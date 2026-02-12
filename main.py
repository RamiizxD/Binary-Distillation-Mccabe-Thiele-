import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------- Helpers ----------------

def x_from_y_left(y_target, x_current, x_eq, y_eq):
    """\
    Horizontal step at y=y_target: find intersection x on piecewise-linear
    equilibrium curve y_eq(x)=y_target.

    Choose the intersection that moves LEFT (x <= x_current) by taking the
    largest feasible x <= x_current.
    """
    y_target = float(y_target)
    x_current = float(x_current)

    xs = []
    for i in range(len(x_eq) - 1):
        x1, x2 = float(x_eq[i]), float(x_eq[i + 1])
        y1, y2 = float(y_eq[i]), float(y_eq[i + 1])

        if (y1 - y_target) * (y2 - y_target) <= 0:
            if abs(y2 - y1) < 1e-15:
                if abs(y1 - y_target) < 1e-12:
                    xs.extend([x1, x2])
            else:
                x_hit = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
                xs.append(x_hit)

    if not xs:
        idx = int(np.argmin(np.abs(np.array(y_eq, dtype=float) - y_target)))
        return float(x_eq[idx])

    xs_left = [x for x in xs if x <= x_current + 1e-12]
    if xs_left:
        return float(max(xs_left))

    return float(min(xs))


def ensure_endpoints(x_eq, y_eq):
    """Ensure curve includes (0,0) and (1,1)."""
    x_eq = np.asarray(x_eq, dtype=float)
    y_eq = np.asarray(y_eq, dtype=float)

    if x_eq[0] > 0.0 + 1e-12:
        x_eq = np.insert(x_eq, 0, 0.0)
        y_eq = np.insert(y_eq, 0, 0.0)
    if x_eq[-1] < 1.0 - 1e-12:
        x_eq = np.append(x_eq, 1.0)
        y_eq = np.append(y_eq, 1.0)

    idx = np.argsort(x_eq)
    return x_eq[idx], y_eq[idx]


def plot_clipped_line(xs, y_line, y_low, y_high, label):
    """Plot only the portion of a line that lies between diagonal and equilibrium."""
    y_line = np.asarray(y_line, dtype=float)
    mask = (y_line >= y_low) & (y_line <= y_high) & np.isfinite(y_line)

    if not np.any(mask):
        return

    idx = np.where(mask)[0]
    starts = [idx[0]]
    ends = []
    for k in range(1, len(idx)):
        if idx[k] != idx[k - 1] + 1:
            ends.append(idx[k - 1])
            starts.append(idx[k])
    ends.append(idx[-1])

    first = True
    for s, e in zip(starts, ends):
        plt.plot(xs[s:e + 1], y_line[s:e + 1], label=label if first else None)
        first = False


def parse_optional(s: str):
    s = (s or "").strip()
    return None if s == "" else float(s)


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="McCabe–Thiele (Excel/Manual VLE)", layout="centered")
st.title("McCabe–Thiele Method")
st.caption("Compute Nmin, Rmin, and actual stages (finite reflux) from VLE data.")

st.header("Inputs")

c1, c2, c3 = st.columns(3)
with c1:
    F = st.number_input("Feed flow rate F", min_value=0.0, value=450.0, step=1.0)
    zL = st.number_input("Light component mol fraction in feed zL", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
with c2:
    xD_in = st.text_input("xD (LK in distillate) — optional", value="0.95")
    xB_in = st.text_input("xB (LK in bottoms) — optional", value="0.05")
with c3:
    D_in = st.text_input("D (distillate flow) — optional", value="")
    B_in = st.text_input("B (bottoms flow) — optional", value="")

xD = parse_optional(xD_in)
xB = parse_optional(xB_in)
D = parse_optional(D_in)
B = parse_optional(B_in)

RRmin = st.number_input("R/Rmin ratio", min_value=1.0, value=1.3, step=0.05)

st.header("Equilibrium (VLE) data")
mode = st.radio("Provide VLE data via:", ["Upload Excel", "Enter manually"], horizontal=True)

uploaded = None
sheet_name = ""
manual_df = None

if mode == "Upload Excel":
    uploaded = st.file_uploader("Upload VLE Excel (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name (blank = first sheet)", value="")
else:
    st.caption("Enter VLE data as two columns: x (liquid) and y (vapor). Add rows as needed.")
    starter = pd.DataFrame({
        "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "y": [0.0, None, None, None, None, None, None, None, None, None, 1.0],
    })
    manual_df = st.data_editor(
        starter,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

run = st.button("Run McCabe–Thiele")

if not run:
    st.info("Enter inputs above, provide VLE data, then click **Run McCabe–Thiele**.")
    st.stop()

# ---------------- Validations ----------------

for name, val in [("zL", zL), ("xD", xD), ("xB", xB)]:
    if val is not None and not (0 <= val <= 1):
        st.error(f"{name} must be between 0 and 1.")
        st.stop()

if mode == "Upload Excel":
    if uploaded is None:
        st.error("Please upload an Excel file with VLE data, or switch to 'Enter manually'.")
        st.stop()
else:
    if manual_df is None or ("x" not in manual_df.columns) or ("y" not in manual_df.columns):
        st.error("Manual VLE table must contain columns 'x' and 'y'.")
        st.stop()
    if manual_df.dropna(subset=["x", "y"]).empty:
        st.error("Manual VLE table is empty. Enter at least a few x/y points.")
        st.stop()

# Overall balance if D or B given
if D is not None and B is None:
    B = F - D
elif B is not None and D is None:
    D = F - B

# Solve D,B if xD & xB provided
if xD is not None and xB is not None:
    if abs(xD - xB) < 1e-12:
        st.error("xD and xB must be different to solve.")
        st.stop()
    D = F * (zL - xB) / (xD - xB)
    B = F - D

# Solve missing xD or xB if D & B known
if D is not None and B is not None:
    if xD is None and xB is not None and abs(D) > 1e-12:
        xD = (F * zL - B * xB) / D
    elif xB is None and xD is not None and abs(B) > 1e-12:
        xB = (F * zL - D * xD) / B

if xD is None or xB is None:
    st.error("Need xD and xB to do McCabe–Thiele stepping.")
    st.stop()
if D is None:
    st.error("Need D (or enough info to compute D) to compute q and Rmin.")
    st.stop()

# ---------------- Load equilibrium (Excel or Manual) ----------------
try:
    if mode == "Upload Excel":
        buf = io.BytesIO(uploaded.getvalue())
        sheet = 0 if sheet_name.strip() == "" else sheet_name.strip()
        raw_df = pd.read_excel(buf, sheet_name=sheet, header=None)

        # Excel format: row0=y, row1=x; skip first column
        y_eq = raw_df.iloc[0, 1:].astype(float).to_numpy()
        x_eq = raw_df.iloc[1, 1:].astype(float).to_numpy()
    else:
        vle = manual_df.dropna(subset=["x", "y"]).astype(float)
        vle = vle[(vle["x"].between(0, 1)) & (vle["y"].between(0, 1))]
        vle = vle.sort_values("x")
        x_eq = vle["x"].to_numpy()
        y_eq = vle["y"].to_numpy()

    idx = np.argsort(x_eq)
    x_eq = x_eq[idx]
    y_eq = y_eq[idx]
    x_eq, y_eq = ensure_endpoints(x_eq, y_eq)
except Exception as e:
    st.exception(e)
    st.stop()

# interpolate y*(x)
def y_equil(x):
    return float(np.interp(float(x), x_eq, y_eq))

# ---------------- Show stream table ----------------
st.subheader("Material balance results")
res_df = pd.DataFrame({
    "Stream": ["Feed", "Distillate", "Bottoms"],
    "Flow":   [F, D, B],
    "Light x": [zL, xD, xB],
    "Heavy x": [1 - zL, 1 - xD, 1 - xB],
})
st.dataframe(res_df, use_container_width=True)

# ---------------- Nmin (total reflux) stepping ----------------
pts_min = [(xD, xD)]
Nmin = 0
x_new, y_new = xD, xD
max_steps = 500
stalled_min = False

while x_new > xB and Nmin < max_steps:
    x_prev = x_new
    x_new = x_from_y_left(y_new, x_prev, x_eq, y_eq)
    pts_min.append((x_new, y_new))
    Nmin += 1

    if abs(x_new - x_prev) < 1e-8:
        stalled_min = True
        break

    if x_new <= xB:
        break

    y_new = x_new
    pts_min.append((x_new, y_new))

fig1 = plt.figure(figsize=(7, 7))
plt.plot(x_eq, y_eq, label="Equilibrium curve")
plt.plot([0, 1], [0, 1], "--", label="Diagonal x = y")
px, py = zip(*pts_min)
plt.plot(px, py, linewidth=2, label="Stepping (Total Reflux)")
plt.axvline(xD, linestyle=":", linewidth=1)
plt.axvline(xB, linestyle=":", linewidth=1)
plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel("x (liquid mole fraction of LK)")
plt.ylabel("y (vapor mole fraction of LK)")
plt.title("Minimum number of Theoretical Stages")
plt.grid(True, alpha=0.3)
plt.legend()
st.pyplot(fig1)
plt.close(fig1)

st.write(f"**Nmin (touches on equilibrium)** = `{Nmin}`" + ("  ⚠️ (stepping stalled)" if stalled_min else ""))

# ---------------- Rmin (pinch with q-line) ----------------
q = 1 - D / F

if abs(q - 1.0) < 1e-12:
    x_pinch = zL
    y_pinch = y_equil(x_pinch)
else:
    def y_q(x):
        return (q/(q-1))*x - zL/(q-1)

    f_vals = y_eq - np.array([y_q(x) for x in x_eq])
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    if len(sign_changes) == 0:
        st.error("Could not find pinch point (q-line does not intersect equilibrium).")
        st.stop()

    i = int(sign_changes[0])
    x1, x2 = x_eq[i], x_eq[i + 1]
    f1, f2 = f_vals[i], f_vals[i + 1]
    x_pinch = x1 - f1 * (x2 - x1) / (f2 - f1)
    y_pinch = y_equil(x_pinch)

m_min = (y_pinch - xD) / (x_pinch - xD)
Rmin = m_min / (1 - m_min)
R = RRmin * Rmin

st.subheader("Reflux results")
st.write(f"Feed quality **q** = `{q:.6f}`")
st.write(f"Pinch point = `(x={x_pinch:.6f}, y={y_pinch:.6f})`")
st.write(f"**Rmin** = `{Rmin:.6f}`")
st.write(f"Using **R/Rmin** = `{RRmin:.3f}` → **R** = `{R:.6f}`")

# ---------------- Finite reflux stepping ----------------
def y_rect(x):
    return (R/(R+1))*x + xD/(R+1)

if abs(q - 1.0) < 1e-12:
    x_int = zL
    y_int = y_rect(x_int)
    def y_q(x):
        return np.nan
else:
    def y_q(x):
        return (q/(q-1))*x - zL/(q-1)

    m_r = R/(R+1)
    b_r = xD/(R+1)
    m_q = q/(q-1)
    b_q = -zL/(q-1)

    x_int = (b_q - b_r) / (m_r - m_q)
    y_int = y_rect(x_int)

m_s = (y_int - xB) / (x_int - xB)
b_s = y_int - m_s * x_int

def y_strip(x):
    return m_s * x + b_s

pts = [(xD, xD)]
N = 0
feed_stage = None
x_new, y_new = xD, xD
stalled = False

while x_new > xB and N < max_steps:
    x_prev = x_new
    x_new = x_from_y_left(y_new, x_prev, x_eq, y_eq)
    pts.append((x_new, y_new))
    N += 1

    if abs(x_new - x_prev) < 1e-8:
        stalled = True
        break

    if x_new <= xB:
        break

    if x_new >= x_int:
        y_new = y_rect(x_new)
    else:
        if feed_stage is None:
            feed_stage = N
        y_new = y_strip(x_new)
    pts.append((x_new, y_new))

fig2 = plt.figure(figsize=(7, 7))
plt.plot(x_eq, y_eq, label="Equilibrium curve")
plt.plot([0, 1], [0, 1], "--", label="Diagonal x = y")

xs = np.linspace(0, 1, 800)
y_diag = xs
y_eq_grid = np.array([y_equil(x) for x in xs])
y_low = np.minimum(y_diag, y_eq_grid)
y_high = np.maximum(y_diag, y_eq_grid)

plot_clipped_line(xs, [y_rect(x) for x in xs], y_low, y_high, label="Rectifying line")
plot_clipped_line(xs, [y_strip(x) for x in xs], y_low, y_high, label="Stripping line")
if abs(q - 1.0) >= 1e-12:
    plot_clipped_line(xs, [y_q(x) for x in xs], y_low, y_high, label="q-line")

px, py = zip(*pts)
plt.plot(px, py, linewidth=2, label="Stepping (finite reflux)")

plt.scatter([xD, xB, x_int], [xD, xB, y_int], s=40)
plt.axvline(xD, linestyle=":", linewidth=1)
plt.axvline(xB, linestyle=":", linewidth=1)
plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel("x (liquid mole fraction of LK)")
plt.ylabel("y (vapor mole fraction of LK)")
plt.title("Actual number of Theoretical Stages (finite reflux)")
plt.grid(True, alpha=0.3)
plt.legend()
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Stage results")
st.write(f"**Actual number of theoretical stages (touches on equilibrium)** = `{N}`" + ("  ⚠️ (stepping stalled)" if stalled else ""))
if feed_stage is not None:
    st.write(f"Estimated feed stage (switch to stripping) ≈ **stage {feed_stage}**")
else:
    st.write("Feed stage not detected (may indicate all steps stayed in rectifying section).")
