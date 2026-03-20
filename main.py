"""
McCabe–Thiele Method — Enhanced
  • VLE data with optional Temperature column
  • Extract VLE from a screenshot using Claude vision
  • Azeotrope detection and T-x-y diagram
"""

import base64
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def x_from_y_left(y_target, x_current, x_eq, y_eq):
    """Horizontal step at y=y_target; find the intersection that moves LEFT."""
    y_target, x_current = float(y_target), float(x_current)
    xs = []
    for i in range(len(x_eq) - 1):
        x1, x2 = float(x_eq[i]), float(x_eq[i + 1])
        y1, y2 = float(y_eq[i]), float(y_eq[i + 1])
        if (y1 - y_target) * (y2 - y_target) <= 0:
            if abs(y2 - y1) < 1e-15:
                if abs(y1 - y_target) < 1e-12:
                    xs.extend([x1, x2])
            else:
                xs.append(x1 + (y_target - y1) * (x2 - x1) / (y2 - y1))
    if not xs:
        idx = int(np.argmin(np.abs(np.array(y_eq, dtype=float) - y_target)))
        return float(x_eq[idx])
    xs_left = [x for x in xs if x <= x_current + 1e-12]
    return float(max(xs_left)) if xs_left else float(min(xs))


def ensure_endpoints(x_eq, y_eq, t_eq=None):
    """Ensure curve includes (0,0) and (1,1)."""
    x_eq = np.asarray(x_eq, dtype=float)
    y_eq = np.asarray(y_eq, dtype=float)
    has_T = t_eq is not None and len(t_eq) == len(x_eq)
    if has_T:
        t_eq = np.asarray(t_eq, dtype=float)

    if x_eq[0] > 1e-12:
        x_eq = np.insert(x_eq, 0, 0.0)
        y_eq = np.insert(y_eq, 0, 0.0)
        if has_T:
            t_eq = np.insert(t_eq, 0, t_eq[0])
    if x_eq[-1] < 1.0 - 1e-12:
        x_eq = np.append(x_eq, 1.0)
        y_eq = np.append(y_eq, 1.0)
        if has_T:
            t_eq = np.append(t_eq, t_eq[-1])

    idx = np.argsort(x_eq)
    if has_T:
        return x_eq[idx], y_eq[idx], t_eq[idx]
    return x_eq[idx], y_eq[idx], None


def detect_azeotropes(x_eq, y_eq, tol=1e-3):
    """Return list of (x, y) where equilibrium curve crosses the diagonal (excl. endpoints)."""
    azeo = []
    diff = y_eq - x_eq
    for i in range(len(diff) - 1):
        if x_eq[i] < tol or x_eq[i + 1] > 1 - tol:
            continue
        if diff[i] * diff[i + 1] <= 0 and abs(diff[i] - diff[i + 1]) > 1e-15:
            xaz = x_eq[i] - diff[i] * (x_eq[i + 1] - x_eq[i]) / (diff[i + 1] - diff[i])
            yaz = np.interp(xaz, x_eq, y_eq)
            azeo.append((float(xaz), float(yaz)))
    return azeo


def plot_clipped_line(xs, y_line, y_low, y_high, **kw):
    y_line = np.asarray(y_line, dtype=float)
    mask = (y_line >= y_low) & (y_line <= y_high) & np.isfinite(y_line)
    if not np.any(mask):
        return
    idx = np.where(mask)[0]
    starts, ends = [idx[0]], []
    for k in range(1, len(idx)):
        if idx[k] != idx[k - 1] + 1:
            ends.append(idx[k - 1]); starts.append(idx[k])
    ends.append(idx[-1])
    first = True
    for s, e in zip(starts, ends):
        plt.plot(xs[s:e + 1], y_line[s:e + 1], **(kw if first else {k: v for k, v in kw.items() if k != "label"}))
        first = False


def parse_optional(s: str):
    s = (s or "").strip()
    return None if s == "" else float(s)


def img_to_b64(file_bytes: bytes) -> str:
    return base64.standard_b64encode(file_bytes).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Claude vision extraction
# ─────────────────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """\
You are a data-extraction assistant for chemical-engineering VLE tables.
The image contains a VLE (Vapor-Liquid Equilibrium) data table.

Extract ALL rows from the table and return ONLY a JSON array — no explanation, no markdown fences.
Each element must be an object with these keys (use null if the column is absent):
  "x"   – liquid-phase mole fraction of the light component (number 0-1)
  "y"   – vapor-phase mole fraction of the light component (number 0-1)
  "T"   – temperature (number, unit as shown; null if not present)

Rules:
• Include the header-row values only if they are numeric.
• Omit any row where x and y cannot be identified.
• Keep the rows in the order they appear in the table.
• Return ONLY the JSON array, nothing else.

Example output:
[{"x":0.0,"y":0.0,"T":100.0},{"x":0.1,"y":0.212,"T":95.5},{"x":1.0,"y":1.0,"T":78.4}]
"""


def extract_vle_from_image(image_bytes: bytes, media_type: str) -> pd.DataFrame:
    """Call Claude API to extract VLE data from a screenshot."""
    try:
        import anthropic
    except ImportError:
        st.error("The `anthropic` package is not installed. Run: pip install anthropic")
        st.stop()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    b64 = img_to_b64(image_bytes)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ],
    )

    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    df = pd.DataFrame(data)
    for col in ["x", "y", "T"]:
        if col not in df.columns:
            df[col] = None
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    return df[["x", "y", "T"]]


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="McCabe–Thiele", layout="centered")
st.title("McCabe–Thiele Method")
st.caption("Compute Nmin, Rmin, and actual stages from VLE data — with temperature and screenshot extraction.")

# ── Process Inputs ────────────────────────────────────────────────────────────
st.header("Process Inputs")
c1, c2, c3 = st.columns(3)
with c1:
    F  = st.number_input("Feed flow rate F",  min_value=0.0, value=450.0, step=1.0)
    zL = st.number_input("Light component mol fraction in feed zL", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
with c2:
    xD_in = st.text_input("xD (LK in distillate) — optional", value="0.95")
    xB_in = st.text_input("xB (LK in bottoms) — optional",   value="0.05")
with c3:
    D_in = st.text_input("D (distillate flow) — optional", value="")
    B_in = st.text_input("B (bottoms flow) — optional",    value="")

xD = parse_optional(xD_in)
xB = parse_optional(xB_in)
D  = parse_optional(D_in)
B  = parse_optional(B_in)

RRmin = st.number_input("R/Rmin ratio", min_value=1.0, value=1.3, step=0.05)

# ── VLE Data ──────────────────────────────────────────────────────────────────
st.header("Equilibrium (VLE) Data")

mode = st.radio(
    "Provide VLE data via:",
    ["Upload Excel", "Enter manually", "Screenshot / image"],
    horizontal=True,
)

uploaded    = None
sheet_name  = ""
manual_df   = None
screenshot  = None
extracted_df: pd.DataFrame | None = None

STARTER = pd.DataFrame({
    "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "y": [0.0, None, None, None, None, None, None, None, None, None, 1.0],
    "T": [None]*11,
})

if mode == "Upload Excel":
    uploaded   = st.file_uploader("Upload VLE Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name (blank = first sheet)", value="")
    st.caption(
        "Expected layout: row 0 = y values, row 1 = x values, "
        "optional row 2 = T values; first column is a label and is skipped."
    )

elif mode == "Enter manually":
    st.caption("Columns: x (liquid), y (vapor), T (temperature — optional).")
    manual_df = st.data_editor(
        STARTER, num_rows="dynamic", use_container_width=True, hide_index=True
    )

else:  # Screenshot
    st.info(
        "Upload a screenshot or photo of a VLE data table. "
        "Claude will extract x, y, and T automatically. "
        "Review and edit the extracted data before running."
    )
    screenshot = st.file_uploader(
        "Upload VLE table screenshot", type=["png", "jpg", "jpeg", "webp"]
    )
    if screenshot is not None:
        col_img, col_btn = st.columns([2, 1])
        with col_img:
            st.image(screenshot, caption="Uploaded screenshot", use_container_width=True)
        with col_btn:
            extract_btn = st.button("🔍 Extract VLE data", use_container_width=True)

        if "extracted_vle" not in st.session_state:
            st.session_state.extracted_vle = None

        if extract_btn:
            with st.spinner("Sending image to Claude for extraction…"):
                ext = screenshot.name.rsplit(".", 1)[-1].lower()
                media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                             "png": "image/png", "webp": "image/webp"}
                media_type = media_map.get(ext, "image/png")
                st.session_state.extracted_vle = extract_vle_from_image(
                    screenshot.getvalue(), media_type
                )
            st.success("Extraction complete — review and edit below.")

        if st.session_state.extracted_vle is not None:
            st.subheader("Extracted VLE data (editable)")
            extracted_df = st.data_editor(
                st.session_state.extracted_vle,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Click **Extract VLE data** to run the extraction.")

# ── Run button ────────────────────────────────────────────────────────────────
run = st.button("▶ Run McCabe–Thiele", type="primary")
if not run:
    st.info("Enter inputs above, provide VLE data, then click **Run McCabe–Thiele**.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

for name, val in [("zL", zL), ("xD", xD), ("xB", xB)]:
    if val is not None and not (0 <= val <= 1):
        st.error(f"{name} must be between 0 and 1.")
        st.stop()

if mode == "Upload Excel" and uploaded is None:
    st.error("Please upload an Excel file or switch mode.")
    st.stop()

if mode == "Enter manually":
    if manual_df is None or manual_df.dropna(subset=["x", "y"]).empty:
        st.error("Enter at least a few x/y points in the VLE table.")
        st.stop()

if mode == "Screenshot / image":
    if extracted_df is None or extracted_df.dropna(subset=["x", "y"]).empty:
        st.error("No VLE data extracted yet. Upload a screenshot and click Extract.")
        st.stop()

# ── Overall balance ───────────────────────────────────────────────────────────
if D is not None and B is None:
    B = F - D
elif B is not None and D is None:
    D = F - B

if xD is not None and xB is not None:
    if abs(xD - xB) < 1e-12:
        st.error("xD and xB must be different.")
        st.stop()
    D = F * (zL - xB) / (xD - xB)
    B = F - D

if D is not None and B is not None:
    if xD is None and xB is not None and abs(D) > 1e-12:
        xD = (F * zL - B * xB) / D
    elif xB is None and xD is not None and abs(B) > 1e-12:
        xB = (F * zL - D * xD) / B

if xD is None or xB is None:
    st.error("Need xD and xB to do McCabe–Thiele stepping.")
    st.stop()
if D is None:
    st.error("Need D (or enough info to compute D).")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load VLE
# ─────────────────────────────────────────────────────────────────────────────
try:
    if mode == "Upload Excel":
        buf   = io.BytesIO(uploaded.getvalue())
        sheet = 0 if sheet_name.strip() == "" else sheet_name.strip()
        raw   = pd.read_excel(buf, sheet_name=sheet, header=None)
        y_eq  = raw.iloc[0, 1:].astype(float).to_numpy()
        x_eq  = raw.iloc[1, 1:].astype(float).to_numpy()
        t_eq  = raw.iloc[2, 1:].astype(float).to_numpy() if len(raw) > 2 else None

    elif mode == "Enter manually":
        vle   = manual_df.dropna(subset=["x", "y"]).copy()
        vle["x"] = pd.to_numeric(vle["x"], errors="coerce")
        vle["y"] = pd.to_numeric(vle["y"], errors="coerce")
        vle   = vle.dropna(subset=["x", "y"])
        vle   = vle[(vle["x"].between(0, 1)) & (vle["y"].between(0, 1))].sort_values("x")
        x_eq  = vle["x"].to_numpy()
        y_eq  = vle["y"].to_numpy()
        t_eq  = vle["T"].to_numpy() if "T" in vle.columns and vle["T"].notna().any() else None

    else:  # Screenshot
        vle   = extracted_df.dropna(subset=["x", "y"]).copy()
        vle["x"] = pd.to_numeric(vle["x"], errors="coerce")
        vle["y"] = pd.to_numeric(vle["y"], errors="coerce")
        vle   = vle.dropna(subset=["x", "y"])
        vle   = vle[(vle["x"].between(0, 1)) & (vle["y"].between(0, 1))].sort_values("x")
        x_eq  = vle["x"].to_numpy()
        y_eq  = vle["y"].to_numpy()
        t_eq  = vle["T"].to_numpy() if "T" in vle.columns and vle["T"].notna().any() else None

    idx  = np.argsort(x_eq)
    x_eq, y_eq = x_eq[idx], y_eq[idx]
    if t_eq is not None:
        t_eq = t_eq[idx]

    x_eq, y_eq, t_eq = ensure_endpoints(x_eq, y_eq, t_eq)

except Exception as e:
    st.exception(e)
    st.stop()


def y_equil(x):
    return float(np.interp(float(x), x_eq, y_eq))


# ─────────────────────────────────────────────────────────────────────────────
# Azeotrope detection
# ─────────────────────────────────────────────────────────────────────────────
azeotropes = detect_azeotropes(x_eq, y_eq)

# ─────────────────────────────────────────────────────────────────────────────
# Material balance
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Material Balance")
res_df = pd.DataFrame({
    "Stream":   ["Feed", "Distillate", "Bottoms"],
    "Flow":     [F, D, B],
    "x (LK)":  [zL, xD, xB],
    "x (HK)":  [1 - zL, 1 - xD, 1 - xB],
})
st.dataframe(res_df, use_container_width=True)

if azeotropes:
    for xaz, yaz in azeotropes:
        st.warning(
            f"⚠️ **Azeotrope detected** at x ≈ {xaz:.4f}, y ≈ {yaz:.4f}. "
            "Separation beyond this point is not possible with simple distillation."
        )
        if xD > xaz + 0.01 or xB < xaz - 0.01:
            st.error(
                "Specified xD or xB crosses the azeotrope composition — "
                "the specified separation cannot be achieved."
            )

# ─────────────────────────────────────────────────────────────────────────────
# Optional T-x-y plot
# ─────────────────────────────────────────────────────────────────────────────
if t_eq is not None:
    st.subheader("T-x-y Diagram")
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    ax0.plot(x_eq, t_eq, "b-o", markersize=4, label="Bubble-point curve (T-x)")
    ax0.plot(y_eq, t_eq, "r-s", markersize=4, label="Dew-point curve (T-y)")
    for xaz, _ in azeotropes:
        taz = float(np.interp(xaz, x_eq, t_eq))
        ax0.axvline(xaz, color="purple", linestyle="--", linewidth=1.2, label=f"Azeotrope x={xaz:.3f}")
        ax0.scatter([xaz], [taz], s=80, color="purple", zorder=5)
    ax0.set_xlabel("Mole fraction of light component")
    ax0.set_ylabel("Temperature")
    ax0.set_title("T-x-y Phase Diagram")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    st.pyplot(fig0)
    plt.close(fig0)

# ─────────────────────────────────────────────────────────────────────────────
# Nmin (total reflux)
# ─────────────────────────────────────────────────────────────────────────────
pts_min = [(xD, xD)]
Nmin    = 0
x_new, y_new = xD, xD
stalled_min  = False
MAX_STEPS    = 500

while x_new > xB and Nmin < MAX_STEPS:
    x_prev = x_new
    x_new  = x_from_y_left(y_new, x_prev, x_eq, y_eq)
    pts_min.append((x_new, y_new))
    Nmin += 1
    if abs(x_new - x_prev) < 1e-8:
        stalled_min = True; break
    if x_new <= xB:
        break
    y_new = x_new
    pts_min.append((x_new, y_new))

fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.plot(x_eq, y_eq, label="Equilibrium curve")
ax1.plot([0, 1], [0, 1], "--", label="Diagonal x = y")
for xaz, yaz in azeotropes:
    ax1.scatter([xaz], [yaz], s=80, color="purple", zorder=5, label=f"Azeotrope ({xaz:.3f})")
px, py = zip(*pts_min)
ax1.plot(px, py, linewidth=2, label="Stepping (Total Reflux)")
ax1.axvline(xD, linestyle=":", linewidth=1)
ax1.axvline(xB, linestyle=":", linewidth=1)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_xlabel("x (liquid mole fraction of LK)")
ax1.set_ylabel("y (vapor mole fraction of LK)")
ax1.set_title("Minimum Number of Theoretical Stages (Total Reflux)")
ax1.grid(True, alpha=0.3)
ax1.legend()
st.subheader("Total Reflux (Nmin)")
st.pyplot(fig1)
plt.close(fig1)
st.write(
    f"**Nmin** = `{Nmin}`" +
    ("  ⚠️ stepping stalled — may be near an azeotrope" if stalled_min else "")
)

# ─────────────────────────────────────────────────────────────────────────────
# Rmin (pinch with q-line)
# ─────────────────────────────────────────────────────────────────────────────
q = 1 - D / F

if abs(q - 1.0) < 1e-12:
    x_pinch = zL
    y_pinch = y_equil(x_pinch)
else:
    def y_q(x):
        return (q / (q - 1)) * x - zL / (q - 1)

    f_vals      = y_eq - np.array([y_q(x) for x in x_eq])
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    if len(sign_changes) == 0:
        st.error("Could not find pinch point (q-line does not intersect equilibrium).")
        st.stop()
    i = int(sign_changes[0])
    x1, x2 = x_eq[i], x_eq[i + 1]
    f1, f2  = f_vals[i], f_vals[i + 1]
    x_pinch = x1 - f1 * (x2 - x1) / (f2 - f1)
    y_pinch = y_equil(x_pinch)

m_min = (y_pinch - xD) / (x_pinch - xD)
Rmin  = m_min / (1 - m_min)
R     = RRmin * Rmin

st.subheader("Reflux Results")
st.write(f"Feed quality **q** = `{q:.6f}`")
st.write(f"Pinch point = `(x={x_pinch:.4f}, y={y_pinch:.4f})`")
st.write(f"**Rmin** = `{Rmin:.4f}`")
st.write(f"Using R/Rmin = {RRmin:.3f} → **R** = `{R:.4f}`")

# ─────────────────────────────────────────────────────────────────────────────
# Finite reflux stepping
# ─────────────────────────────────────────────────────────────────────────────
def y_rect(x):
    return (R / (R + 1)) * x + xD / (R + 1)

if abs(q - 1.0) < 1e-12:
    x_int = zL
    y_int = y_rect(x_int)
    def y_q_line(x): return np.nan
else:
    def y_q_line(x):
        return (q / (q - 1)) * x - zL / (q - 1)

    m_r, b_r = R / (R + 1), xD / (R + 1)
    m_q, b_q = q / (q - 1), -zL / (q - 1)
    x_int    = (b_q - b_r) / (m_r - m_q)
    y_int    = y_rect(x_int)

m_s = (y_int - xB) / (x_int - xB)
b_s = y_int - m_s * x_int

def y_strip(x):
    return m_s * x + b_s

pts        = [(xD, xD)]
N          = 0
feed_stage = None
x_new, y_new = xD, xD
stalled    = False

while x_new > xB and N < MAX_STEPS:
    x_prev = x_new
    x_new  = x_from_y_left(y_new, x_prev, x_eq, y_eq)
    pts.append((x_new, y_new))
    N += 1
    if abs(x_new - x_prev) < 1e-8:
        stalled = True; break
    if x_new <= xB:
        break
    if x_new >= x_int:
        y_new = y_rect(x_new)
    else:
        if feed_stage is None:
            feed_stage = N
        y_new = y_strip(x_new)
    pts.append((x_new, y_new))

xs         = np.linspace(0, 1, 800)
y_diag     = xs
y_eq_grid  = np.array([y_equil(x) for x in xs])
y_low      = np.minimum(y_diag, y_eq_grid)
y_high     = np.maximum(y_diag, y_eq_grid)

fig2, ax2  = plt.subplots(figsize=(7, 7))
ax2.plot(x_eq, y_eq, label="Equilibrium curve")
ax2.plot([0, 1], [0, 1], "--", label="Diagonal x = y")

for xaz, yaz in azeotropes:
    ax2.scatter([xaz], [yaz], s=80, color="purple", zorder=5, label=f"Azeotrope ({xaz:.3f})")
    ax2.axvline(xaz, color="purple", linestyle=":", linewidth=1)

plot_clipped_line(xs, [y_rect(x) for x in xs],  y_low, y_high, label="Rectifying line",  color="tab:orange")
plot_clipped_line(xs, [y_strip(x) for x in xs], y_low, y_high, label="Stripping line",   color="tab:green")
if abs(q - 1.0) >= 1e-12:
    plot_clipped_line(xs, [y_q_line(x) for x in xs], y_low, y_high, label="q-line", color="tab:red", linestyle="--")

px, py = zip(*pts)
ax2.plot(px, py, linewidth=2, label="Stepping (finite reflux)", color="tab:blue")
ax2.scatter([xD, xB, x_int], [xD, xB, y_int], s=50, color="k", zorder=5)
ax2.axvline(xD, linestyle=":", linewidth=1)
ax2.axvline(xB, linestyle=":", linewidth=1)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_xlabel("x (liquid mole fraction of LK)")
ax2.set_ylabel("y (vapor mole fraction of LK)")
ax2.set_title("Actual Theoretical Stages (Finite Reflux)")
ax2.grid(True, alpha=0.3)
ax2.legend()

st.subheader("Finite Reflux Stepping")
st.pyplot(fig2)
plt.close(fig2)

st.write(
    f"**Actual theoretical stages** = `{N}`" +
    ("  ⚠️ stepping stalled — may be at or beyond an azeotrope" if stalled else "")
)
if feed_stage is not None:
    st.write(f"Estimated optimal feed stage ≈ **stage {feed_stage}**")
else:
    st.write("Feed stage not detected (all steps in rectifying section).")

# ── Summary table ─────────────────────────────────────────────────────────────
st.subheader("Summary")
summary = {
    "Parameter": ["Nmin", "Rmin", "R (actual)", "N (actual)", "Feed stage", "Azeotropes"],
    "Value": [
        str(Nmin),
        f"{Rmin:.4f}",
        f"{R:.4f}",
        str(N),
        str(feed_stage) if feed_stage else "—",
        ", ".join(f"x={xa:.4f}" for xa, _ in azeotropes) if azeotropes else "None detected",
    ],
}
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
