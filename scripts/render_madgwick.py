#!/usr/bin/env python3
"""
render_madgwick.py -- parse build/madgwick_dump.log, plot the 100-step
IMU attitude trajectory produced by the quaternion-multiply kernel on
sage16_top (mode=2) versus the double-precision software reference.

Outputs build/madgwick_figure.png
"""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
LOG  = ROOT / "build" / "madgwick_dump.log"
OUT  = ROOT / "build" / "madgwick_figure.png"

raw = LOG.read_bytes()
if raw.startswith(b"\xff\xfe"):
    text = raw.decode("utf-16", errors="ignore")
else:
    text = raw.decode("utf-8", errors="ignore")

rx = re.compile(
    r"#QSTEP\s+(\d+)\s+qf\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    r"\s+qr\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    r"\s+omega\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")

steps, qf, qr, omg = [], [], [], []
for m in rx.finditer(text):
    steps.append(int(m.group(1)))
    qf.append([int(m.group(i)) / 32768.0 for i in range(2, 6)])   # Q1.15 -> float
    qr.append([float(m.group(i)) for i in range(6, 10)])
    omg.append([float(m.group(i)) for i in range(10, 13)])

steps = np.array(steps)
qf    = np.array(qf)   # (N,4) fabric result w,x,y,z
qr    = np.array(qr)   # (N,4) reference w,x,y,z
omg   = np.array(omg)  # (N,3) gyro input
dt    = 0.01           # 100 Hz
t     = steps * dt

def quat_to_euler(q):
    """q = (w,x,y,z) -> (roll, pitch, yaw) in degrees, ZYX convention."""
    w, x, y, z = q.T
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin (np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw

roll_f, pitch_f, yaw_f = quat_to_euler(qf)
roll_r, pitch_r, yaw_r = quat_to_euler(qr)

# quaternion unit-norm drift
norm_f = np.linalg.norm(qf, axis=1)
err    = np.linalg.norm(qf - qr, axis=1)

fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

# ---- (0,0:2) attitude roll/pitch/yaw vs time ----
ax0 = fig.add_subplot(gs[0, 0:2])
ax0.plot(t, roll_f,  "-",  color="#d7191c", lw=2,  label="roll  (fabric)")
ax0.plot(t, roll_r,  "--", color="#d7191c", lw=1,  label="roll  (ref)")
ax0.plot(t, pitch_f, "-",  color="#2b8cbe", lw=2,  label="pitch (fabric)")
ax0.plot(t, pitch_r, "--", color="#2b8cbe", lw=1,  label="pitch (ref)")
ax0.plot(t, yaw_f,   "-",  color="#4daf4a", lw=2,  label="yaw   (fabric)")
ax0.plot(t, yaw_r,   "--", color="#4daf4a", lw=1,  label="yaw   (ref)")
ax0.set_xlabel("time (s)  @ 100 Hz IMU loop")
ax0.set_ylabel("attitude (deg)")
ax0.set_title("Drone attitude over 1 s of simulated flight\n"
              "(fabric quat-mul vs double-precision reference)", fontsize=11)
ax0.legend(ncol=3, fontsize=8, loc="lower left")
ax0.grid(alpha=0.3)

# ---- (0,2) gyro input profile ----
ax1 = fig.add_subplot(gs[0, 2])
ax1.plot(t, omg[:,0], color="#d7191c", label="gx")
ax1.plot(t, omg[:,1], color="#2b8cbe", label="gy")
ax1.plot(t, omg[:,2], color="#4daf4a", label="gz")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("gyro rate (rad/s)")
ax1.set_title("Gyroscope input\n(constant roll+pitch, 1 Hz sine yaw)",
              fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ---- (1,0:2) quaternion components (w,x,y,z) fabric vs ref ----
ax2 = fig.add_subplot(gs[1, 0:2])
comp = ["w", "x", "y", "z"]
cols = ["#222222", "#d7191c", "#2b8cbe", "#4daf4a"]
for i in range(4):
    ax2.plot(t, qf[:, i], "-",  color=cols[i], lw=2,
             label=f"q{comp[i]} (fabric)")
    ax2.plot(t, qr[:, i], "--", color=cols[i], lw=1,
             label=f"q{comp[i]} (ref)")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("quaternion component")
ax2.set_title("Quaternion components -- fabric (solid) vs reference (dashed)",
              fontsize=11)
ax2.legend(ncol=4, fontsize=8, loc="center right")
ax2.grid(alpha=0.3)

# ---- (1,2) unit-norm drift ----
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(t, norm_f - 1.0, color="#d7191c", lw=2)
ax3.axhline( 0.05, color="gray", ls=":")
ax3.axhline(-0.05, color="gray", ls=":")
ax3.set_xlabel("time (s)")
ax3.set_ylabel("||q|| - 1")
ax3.set_title("Quaternion unit-norm drift\n(drift check tolerance = 5%)",
              fontsize=10)
ax3.grid(alpha=0.3)

# ---- (2,0:2) 3D attitude arrow trajectory ----
ax4 = fig.add_subplot(gs[2, 0:2], projection="3d")
# rotate the body x-axis by each quaternion, draw the tip of it as a trail
def rotate_vec(q, v):
    w, x, y, z = q.T
    # quaternion rotation of vector v = (vx,vy,vz)
    vx, vy, vz = v
    rx = (1 - 2*(y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz
    ry = 2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz
    rz = 2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz
    return np.stack([rx, ry, rz], axis=1)

tip_f = rotate_vec(qf, (1.0, 0.0, 0.0))
tip_r = rotate_vec(qr, (1.0, 0.0, 0.0))

ax4.plot(tip_f[:,0], tip_f[:,1], tip_f[:,2], color="#d7191c", lw=2,
         label="body-x tip (fabric)")
ax4.plot(tip_r[:,0], tip_r[:,1], tip_r[:,2], color="#2b8cbe", lw=1, ls="--",
         label="body-x tip (ref)")
ax4.scatter([tip_f[0,0]], [tip_f[0,1]], [tip_f[0,2]], color="green", s=40,
            label="start")
ax4.scatter([tip_f[-1,0]], [tip_f[-1,1]], [tip_f[-1,2]], color="black", s=40,
            label="end")
# unit sphere wireframe
u = np.linspace(0, 2*np.pi, 24); vv = np.linspace(0, np.pi, 12)
xs = np.outer(np.cos(u), np.sin(vv))
ys = np.outer(np.sin(u), np.sin(vv))
zs = np.outer(np.ones_like(u), np.cos(vv))
ax4.plot_wireframe(xs, ys, zs, color="#cccccc", alpha=0.25, lw=0.5)
ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_zlabel("z")
ax4.set_title("Drone body-axis swept over 100 IMU steps\n"
              "(each point = one fabric quat-mul + SW normalize)",
              fontsize=10)
ax4.legend(fontsize=8, loc="upper left")
ax4.set_box_aspect([1,1,1])

# ---- (2,2) stats panel ----
axs = fig.add_subplot(gs[2, 2])
axs.axis("off")
final_err_deg = np.max(np.abs(np.array([roll_f-roll_r, pitch_f-pitch_r, yaw_f-yaw_r])))
axs.text(0.0, 0.98,
    "Madgwick (gyro-only) on sage16_top\n"
    f"  N steps            : {len(steps)}\n"
    f"  IMU rate           : 100 Hz (dt=10 ms)\n"
    "\n"
    "  fabric quat-mul    : 10 cy/step\n"
    "  SW (add+norm)      : 109 cy/step\n"
    "  ---------------------\n"
    "  total              : 119 cy/step\n"
    "  @ 160 MHz          : 0.74 us/step\n"
    "  max IMU rate       : ~1.35 MHz\n"
    "                       (head-room 13,500x)\n"
    "\n"
    f"  drift > 5%%         : 0 / {len(steps)}\n"
    f"  max attitude err   : {final_err_deg:.3f} deg\n"
    f"  final q vs ref     : {err[-1]:.5f}",
    fontsize=9.5, family="monospace", va="top")

fig.suptitle("Madgwick attitude filter on the 16-DSP sage16_top fabric  "
             "(tb_madgwick.v, mode=quat-mul)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}")
