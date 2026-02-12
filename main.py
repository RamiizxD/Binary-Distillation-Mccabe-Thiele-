import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Helpers ----------------

def get_float_optional(prompt: str):
    s = input(prompt + " (press Enter to skip): ").strip()
    return None if s == "" else float(s)


def x_from_y_left(y_target, x_current, x_eq, y_eq):
    """\
    Given a horizontal step at y=y_target, find the intersection x on the
    piecewise-linear equilibrium curve y_eq(x)=y_target.

    Choose the intersection that moves LEFT (x <= x_current) by taking the
    largest feasible x <= x_current.

    This prevents stalling/oscillation that can happen when inverting y(x)
    via simple interpolation on y-sorted data.
    """
    y_target = float(y_target)
    x_current = float(x_current)

    xs = []
    for i in range(len(x_eq) - 1):
        x1, x2 = float(x_eq[i]), float(x_eq[i + 1])
        y1, y2 = float(y_eq[i]), float(y_eq[i + 1])

        # If y_target is between y1 and y2 (inclusive), segment crosses y_target
        if (y1 - y_target) * (y2 - y_target) <= 0:
            if abs(y2 - y1) < 1e-15:
                # Flat segment: if exactly on y_target, accept endpoints
                if abs(y1 - y_target) < 1e-12:
                    xs.extend([x1, x2])
            else:
                # Linear interpolation for x at y_target
                x_hit = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
                xs.append(x_hit)

    if not xs:
        # Fallback: nearest y point
        idx = int(np.argmin(np.abs(np.array(y_eq, dtype=float) - y_target)))
        return float(x_eq[idx])

    xs_left = [x for x in xs if x <= x_current + 1e-12]
    if xs_left:
        return float(max(xs_left))

    # If nothing is left (can happen at very top), choose the smallest solution
    return float(min(xs))


def ensure_endpoints(x_eq, y_eq):
    """Ensure curve includes (0,0) and (1,1) to avoid endpoint clipping issues."""
    x_eq = np.asarray(x_eq, dtype=float)
    y_eq = np.asarray(y_eq, dtype=float)

    if x_eq[0] > 0.0 + 1e-12:
        x_eq = np.insert(x_eq, 0, 0.0)
        y_eq = np.insert(y_eq, 0, 0.0)
    if x_eq[-1] < 1.0 - 1e-12:
        x_eq = np.append(x_eq, 1.0)
        y_eq = np.append(y_eq, 1.0)

    # sort by x again
    idx = np.argsort(x_eq)
    return x_eq[idx], y_eq[idx]


# ---------------- Inputs ----------------

print("Please enter the following data (binary distillation)")
F = float(input("Feed flow rate F: "))
zL = float(input("Light component mol fraction in feed zL: "))

xD = get_float_optional("Light component mol fraction in distillate xD")
xB = get_float_optional("Light component mol fraction in bottoms xB")
D = get_float_optional("Distillate flow rate D")
B = get_float_optional("Bottoms flow rate B")

# Validate fractions if provided
for name, val in [("zL", zL), ("xD", xD), ("xB", xB)]:
    if val is not None and not (0 <= val <= 1):
        raise ValueError(f"{name} must be between 0 and 1.")

# Overall balance if D or B given
if D is not None and B is None:
    B = F - D
elif B is not None and D is None:
    D = F - B

# Solve D,B if xD & xB provided
if xD is not None and xB is not None:
    if abs(xD - xB) < 1e-12:
        raise ValueError("xD and xB must be different to solve.")
    D = F * (zL - xB) / (xD - xB)
    B = F - D

# Solve missing xD or xB if D & B known
if D is not None and B is not None:
    if xD is None and xB is not None and abs(D) > 1e-12:
        xD = (F * zL - B * xB) / D
    elif xB is None and xD is not None and abs(B) > 1e-12:
        xB = (F * zL - D * xD) / B

print("\n--- Results ---")
print(f"Distillate flow D = {D}")
print(f"Bottoms flow    B = {B}")
print(f"Distillate LK fraction xD = {xD}")
print(f"Bottoms   LK fraction xB = {xB}")

if xD is None or xB is None:
    raise ValueError("Need xD and xB to do McCabe–Thiele stepping.")
if D is None:
    raise ValueError("Need D (or enough info to compute D) to compute q and Rmin.")

# ---------------- Load equilibrium (Excel: row0=y, row1=x; skip first col) ----------------
excel_path = input("\nEquilibrium Excel file path (e.g., VLE.xlsx): ").strip().strip('"').strip("'")
sheet_name = input("Sheet name (press Enter for first sheet): ").strip() or 0
raw_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

y_eq = raw_df.iloc[0, 1:].astype(float).to_numpy()
x_eq = raw_df.iloc[1, 1:].astype(float).to_numpy()

# sort by x
idx = np.argsort(x_eq)
x_eq = x_eq[idx]
y_eq = y_eq[idx]

# add endpoints if needed
x_eq, y_eq = ensure_endpoints(x_eq, y_eq)

# interpolate y*(x)
def y_equil(x):
    return float(np.interp(float(x), x_eq, y_eq))

# ---------------- Nmin (total reflux) stepping ----------------
pts_min = [(xD, xD)]
Nmin = 0
x_new, y_new = xD, xD
max_steps = 500

while x_new > xB and Nmin < max_steps:
    x_prev = x_new
    x_new = x_from_y_left(y_new, x_prev, x_eq, y_eq)  # horizontal to equilibrium
    pts_min.append((x_new, y_new))
    Nmin += 1

    if abs(x_new - x_prev) < 1e-8:
        print("Nmin stepping stalled; check equilibrium data/endpoints.")
        break

    if x_new <= xB:
        break

    y_new = x_new  # vertical to diagonal
    pts_min.append((x_new, y_new))

plt.figure(figsize=(7, 7))
plt.plot(x_eq, y_eq, label="Equilibrium curve (Excel)")
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
plt.show()
print(f"\nNmin (touches on equilibrium) = {Nmin}")

# ---------------- Rmin (pinch with q-line) ----------------
q = 1 - D / F
print(f"Feed quality q = {q:.6f}")

if abs(q - 1.0) < 1e-12:
    # q=1 => x=zL vertical; pinch at x=zL
    x_pinch = zL
    y_pinch = y_equil(x_pinch)
else:
    def y_q(x):
        return (q/(q-1))*x - zL/(q-1)

    f_vals = y_eq - np.array([y_q(x) for x in x_eq])
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    if len(sign_changes) == 0:
        raise RuntimeError("Could not find pinch point (q-line does not intersect equilibrium).")

    i = int(sign_changes[0])
    x1, x2 = x_eq[i], x_eq[i+1]
    f1, f2 = f_vals[i], f_vals[i+1]
    x_pinch = x1 - f1*(x2-x1)/(f2-f1)
    y_pinch = y_equil(x_pinch)

m_min = (y_pinch - xD) / (x_pinch - xD)
Rmin = m_min / (1 - m_min)
print(f"Pinch point: x={x_pinch:.6f}, y={y_pinch:.6f}")
print(f"Rmin = {Rmin:.6f}")

# ---------------- Finite reflux stepping ----------------
RRmin = float(input("\nEnter the R/Rmin ratio (e.g., 1.3): "))
R = RRmin * Rmin
print(f"Using R = {R:.6f}")

# rectifying line
def y_rect(x):
    return (R/(R+1))*x + xD/(R+1)

# q-line and intersection with rectifying line
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

# stripping line through (xB,xB) and (x_int,y_int)
m_s = (y_int - xB) / (x_int - xB)
b_s = y_int - m_s * x_int

def y_strip(x):
    return m_s * x + b_s

# step between equilibrium and operating lines
pts = [(xD, xD)]
N = 0
feed_stage = None
x_new, y_new = xD, xD

while x_new > xB and N < max_steps:
    x_prev = x_new
    x_new = x_from_y_left(y_new, x_prev, x_eq, y_eq)  # horizontal to equilibrium
    pts.append((x_new, y_new))
    N += 1

    if abs(x_new - x_prev) < 1e-8:
        print("Finite reflux stepping stalled; check R/Rmin and curve endpoints.")
        break

    if x_new <= xB:
        break

    # vertical to correct operating line
    if x_new >= x_int:
        y_new = y_rect(x_new)
    else:
        if feed_stage is None:
            feed_stage = N
        y_new = y_strip(x_new)

    pts.append((x_new, y_new))

# plot finite reflux
plt.figure(figsize=(7, 7))
plt.plot(x_eq, y_eq, label="Equilibrium curve (Excel)")
plt.plot([0, 1], [0, 1], "--", label="Diagonal x = y")

xs = np.linspace(0, 1, 800)

y_diag = xs
# equilibrium y*(x) on the same grid
y_eq_grid = np.array([y_equil(x) for x in xs])
# bounds of the region between equilibrium curve and diagonal
y_low = np.minimum(y_diag, y_eq_grid)
y_high = np.maximum(y_diag, y_eq_grid)

def plot_clipped_line(y_line, label):
    """Plot only the portion of a line that lies between the diagonal and equilibrium curve."""
    y_line = np.asarray(y_line, dtype=float)
    mask = (y_line >= y_low) & (y_line <= y_high) & np.isfinite(y_line)

    # break into contiguous segments so matplotlib doesn't connect across gaps
    if not np.any(mask):
        return
    idx = np.where(mask)[0]
    starts = [idx[0]]
    ends = []
    for k in range(1, len(idx)):
        if idx[k] != idx[k-1] + 1:
            ends.append(idx[k-1])
            starts.append(idx[k])
    ends.append(idx[-1])

    for s, e in zip(starts, ends):
        plt.plot(xs[s:e+1], y_line[s:e+1], label=label)
        label = None  # only show label once in legend

# compute line values
plot_clipped_line([y_rect(x) for x in xs], label="Rectifying line")
plot_clipped_line([y_strip(x) for x in xs], label="Stripping line")
if abs(q - 1.0) >= 1e-12:
    plot_clipped_line([y_q(x) for x in xs], label="q-line")

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
plt.show()

print(f"\nActual number of theoretical stages (touches on equilibrium) = {N}")
if feed_stage is not None:
    print(f"Estimated feed stage (switch to stripping) ≈ stage {feed_stage}")
else:
    print("Feed stage not detected (may indicate all steps stayed in rectifying section).")
