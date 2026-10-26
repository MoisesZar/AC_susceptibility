#!/usr/bin/env python
# coding: utf-8
"""
AC susceptibility of nanoparticle assemblies (micromagnetic).

Computes the real and imaginary parts of the AC susceptibility of an
assembly of magnetic nanoparticles as a function of temperature.

Each simulation cell represents a nanoparticle with:
  - uniaxial anisotropy (random easy axis per cell)
  - dipolar interactions (optional, controlled by the packing fraction)
  - an external AC magnetic field

Container shapes and packing fractions are selectable.
Built on Ubermag / OOMMF.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.fft import fft, ifft, fftfreq
from tqdm import tqdm

import psutil
import threading
import time

import oommfc as mc
import discretisedfield as df
import micromagneticmodel as mm


# ----------------------------------------------------------------------
# Physical and numerical parameters
# ----------------------------------------------------------------------
n0 = 0.8388e-9                  # reference length (m)
K = 1.1e4                       # uniaxial anisotropy constant (J/m^3)
rho = 5170                      # density (kg/m^3)
MW = 234.37                     # molar mass (g/mol)
Na = 6.022e23                   # Avogadro constant
n = Na * rho * 1000 / MW        # number density (1/m^3)
Ms = 2.2 * mm.consts.muB * n    # saturation magnetization (A/m)
alpha = 0.05                    # Gilbert damping

f = 50e6                        # AC field frequency (Hz)
omega = 2 * np.pi * f
ciclos = 1                      # number of AC cycles per temperature
dt = 1e-12                      # timestep (s)
H0 = 10e-3 / mm.consts.mu0      # AC field amplitude (A/m)
D = 6                           # particle diameter in units of n0
phi = 1 / 8                     # packing fraction: 1, 1/2, 1/8 or 0

# Temperature sweep: coarser at high T, finer at low T
T_values = np.concatenate((
    np.arange(400, 200, -50),
    np.arange(200, 100, -10),
    np.arange(100, 50, -5),
    np.arange(50, 7.5, -2.5),
))

# Reproducibility (comment out if you want fresh random configurations)
np.random.seed(42)

dNP = D * n0                                  # particle size (m)
phi_aux = 1 if phi == 0 else phi              # avoid division by zero
d = dNP * int(16 / phi_aux ** (1 / 3))        # box size, scales with phi

system = mm.System(name='AC_susceptibility_shapes')


# ----------------------------------------------------------------------
# Geometry: shape functions
# Each function returns Ms inside the shape and 0 outside.
# The packing fraction phi decides which cells are magnetic.
# ----------------------------------------------------------------------
def _packing_ok(x, y, z):
    """Return True if the cell at (x, y, z) is magnetic for this phi."""
    if phi in (0, 1):
        return True
    if phi == 0.5:
        # 50% filling: checkerboard-like pattern
        return (x + y + z + (36 + 1) ** 3 * 0.5) % 2 == 0
    if phi == 1 / 8:
        # 12.5% filling: sparse pattern
        return (
            (x + 0.5 + 60 * 2) % 2 != 0
            and (y + 0.5 + 60 * 2) % 2 != 0
            and (z + 0.5 + 60 * 2) % 2 != 0
        )
    return False


def Ms_value_sph(pos):
    """Solid sphere."""
    x, y, z = pos / dNP
    inside = x**2 + y**2 + z**2 <= (d / dNP / 2) ** 2
    return Ms if inside and _packing_ok(x, y, z) else 0


def Ms_value_spheroid_op(pos):
    """Spheroid, out-of-plane (flattened along z)."""
    x, y, z = pos / dNP
    inside = (
        (x / (d / dNP / 2)) ** 2
        + (y / (d / dNP / 2)) ** 2
        + (z / (d / dNP / 8)) ** 2
        <= 1
    )
    return Ms if inside and _packing_ok(x, y, z) else 0


def Ms_value_spheroid_ip(pos):
    """Spheroid, in-plane (flattened along x)."""
    x, y, z = pos / dNP
    inside = (
        (x / (d / dNP / 8)) ** 2
        + (y / (d / dNP / 2)) ** 2
        + (z / (d / dNP / 2)) ** 2
        <= 1
    )
    return Ms if inside and _packing_ok(x, y, z) else 0


def Ms_value_disc_ip(pos):
    """Disc, in-plane (thin along x, circular in the yz plane)."""
    x, y, z = pos / dNP
    inside = z**2 + y**2 <= (d / dNP / 2) ** 2 and abs(x) <= d / dNP / 8
    return Ms if inside and _packing_ok(x, y, z) else 0


def Ms_value_disc_op(pos):
    """Disc, out-of-plane (thin along z, circular in the xy plane)."""
    x, y, z = pos / dNP
    inside = x**2 + y**2 <= (d / dNP / 2) ** 2 and abs(z) <= d / dNP / 8
    return Ms if inside and _packing_ok(x, y, z) else 0


def Ms_value_sq(pos):
    """Solid cube: the whole box is magnetic."""
    x, y, z = pos / dNP
    return Ms if _packing_ok(x, y, z) else 0


def Ms_value_rect_op(pos):
    """Rectangular slab, out-of-plane (thin along z)."""
    x, y, z = pos / dNP
    inside = abs(z) <= d / dNP / 8
    return Ms if inside and _packing_ok(x, y, z) else 0


def Ms_value_rect_ip(pos):
    """Rectangular slab, in-plane (thin along x)."""
    x, y, z = pos / dNP
    inside = abs(x) <= d / dNP / 8
    return Ms if inside and _packing_ok(x, y, z) else 0


# ----------------------------------------------------------------------
# Mesh and fields
# Change the shape function used in `norm=` to switch geometry.
# Default: cube (Ms_value_sq).
# ----------------------------------------------------------------------
cell = (dNP, dNP, dNP)
region = df.Region(p1=(-d / 2, -d / 2, -d / 2), p2=(d / 2, d / 2, d / 2))
mesh = df.Mesh(region=region, cell=cell)

system.m = df.Field(
    mesh,
    nvdim=3,
    value=lambda pos: np.random.uniform(-1, 1, size=3),
    norm=Ms_value_sq,          # <- swap shape here
    valid="norm",
)

# Random uniaxial easy axis per cell
u = df.Field(mesh, nvdim=3, value=lambda pos: np.random.uniform(-1, 1, size=3))


# ----------------------------------------------------------------------
# Energies and dynamics
# ----------------------------------------------------------------------
def cos_field(t):
    """AC field: cos(omega * t)."""
    return np.cos(omega * t)


system.energy = (
    mm.UniaxialAnisotropy(K=K, u=u)
    + mm.Zeeman(H=(0, 0, H0), func=cos_field, dt=dt)
)
if phi != 0:
    # Dipolar interactions are switched off for the non-interacting case phi = 0
    system.energy += mm.Demag()

system.dynamics = (
    mm.Damping(alpha=alpha)
    + mm.Precession(gamma0=mm.consts.gamma0)
)


# ----------------------------------------------------------------------
# Optional: snapshot of the initial magnetization
# ----------------------------------------------------------------------
def snapshot(title):
    """Plot Mz on the xy plane and Mx on the yz plane."""
    plane_xy = system.m.sel('z')
    plane_yz = system.m.sel('x')

    fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

    plane_xy.z.mpl(
        ax=axs[0],
        scalar_kw={
            "colorbar_label": "Mz (A/m)",
            "cmap": "cividis",
            "symmetric_clim": True,
        },
        vector_kw={"color": "white"},
    )
    plane_yz.mpl(
        ax=axs[1],
        scalar_kw={
            "colorbar_label": "Mx (A/m)",
            "cmap": "cividis",
            "symmetric_clim": True,
        },
        vector_kw={"color": "white"},
    )
    axs[0].set_title("xy plane")
    axs[1].set_title("yz plane")
    fig.suptitle(title, fontsize="xx-large")
    fig.tight_layout()


# snapshot("Initial state")   # uncomment to inspect the initial state


# ----------------------------------------------------------------------
# Optional: raise the priority of a newly launched oxs.exe (Windows only)
# ----------------------------------------------------------------------
def monitor_oxs_realtime(timeout=10, delay=0.2):
    """
    Watch for a new `oxs.exe` process and set it to real-time priority.
    Useful for long OOMMF runs on Windows. Runs in a background thread.
    """
    def _monitor():
        existing_pids = {
            p.pid
            for p in psutil.process_iter(['name'])
            if p.info.get('name') == 'oxs.exe'
        }
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                current = [
                    p
                    for p in psutil.process_iter(['name'])
                    if p.info.get('name') == 'oxs.exe'
                ]
                for p in current:
                    if p.pid not in existing_pids:
                        try:
                            p.nice(psutil.REALTIME_PRIORITY_CLASS)
                            print(f"oxs.exe (PID {p.pid}) set to real-time priority.")
                            return
                        except Exception as e:
                            print(f"Could not change priority of PID {p.pid}: {e}")
                time.sleep(delay)
            except Exception as e:
                print(f"Monitor error: {e}")
                break

    threading.Thread(target=_monitor, daemon=True).start()


# ----------------------------------------------------------------------
# Susceptibility from the simulated time-dependent magnetization
# ----------------------------------------------------------------------
def filtered_magnetization(M, t):
    """
    Band-pass FFT filter around the excitation frequency f.
    Removes transient and higher-harmonic contributions.
    """
    N = len(t)
    M_f = fft(M)
    freqs = fftfreq(N, t[1] - t[0])
    bandwidth = 0.1 * f
    mask = (np.abs(freqs) > (f - bandwidth)) & (np.abs(freqs) < (f + bandwidth))
    M_f_filtered = np.zeros_like(M_f)
    M_f_filtered[mask] = M_f[mask]
    return ifft(M_f_filtered).real


def chi_real(t, M, B):
    """Real part of the AC susceptibility from one field cycle."""
    return 2 * f / (mm.consts.mu0 * H0**2) / ciclos * simpson(B * M, x=t)


def chi_imag(t, M):
    """Imaginary part of the AC susceptibility from one field cycle."""
    return 2 * f / H0 / ciclos * simpson(M * np.sin(omega * t), x=t)


# ----------------------------------------------------------------------
# Main loop: temperature sweep
# ----------------------------------------------------------------------
chi_r = {"integral": [], "integral filtered": []}
chi_c = {"integral": [], "integral filtered": []}

for T in tqdm(T_values, desc="Computing chi(omega)"):
    system.T = T

    evolver = mc.UHH_ThetaEvolver(fixed_timestep=dt, alpha=alpha)
    td = mc.TimeDriver(evolver=evolver)

    # Optional: raise OOMMF priority (Windows only)
    # monitor_oxs_realtime(timeout=10)

    td.drive(system, t=1 / f, n=100)

    t = system.table.data['t'].values
    m = Ms * system.table.data['mz'].values
    B = system.table.data['Bz_zeeman'].values * 1e-3

    m_filtered = filtered_magnetization(m, t)

    chi_r["integral"].append(chi_real(t, m, B))
    chi_r["integral filtered"].append(chi_real(t, m_filtered, B))
    chi_c["integral"].append(chi_imag(t, m))
    chi_c["integral filtered"].append(chi_imag(t, m_filtered))

    print(
        f"T = {T:6.1f} K | "
        f"chi_r = {chi_r['integral'][-1]:.4e} "
        f"(filtered {chi_r['integral filtered'][-1]:.4e}) | "
        f"chi_c = {chi_c['integral'][-1]:.4e} "
        f"(filtered {chi_c['integral filtered'][-1]:.4e})"
    )


# ----------------------------------------------------------------------
# Plots (simulation only)
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(
    T_values[1:],
    np.array(chi_r["integral filtered"][1:]),
    'd',
    label='Simulation (filtered)',
)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel(r'$\chi_r$')
ax.set_title(r'$\chi_r$ as a function of T')
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.plot(
    T_values[1:],
    np.array(chi_c["integral filtered"][1:]),
    'd',
    label='Simulation (filtered)',
)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel(r'$\chi_c$')
ax.set_title(r'$\chi_c$ as a function of T')
ax.legend()
ax.grid(True)
plt.show()


# ----------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------
suffix = (
    f"_f_{f * 1e-6:.0f}MHz"
    f"_D_{D * n0 * 1e9:.1f}nm"
    f"_H_{H0 * mm.consts.mu0 * 1e3:.1f}mT"
    f"_phi_{phi}"
)

data_r = np.column_stack((
    T_values,
    chi_r["integral"],
    chi_r["integral filtered"],
))
np.savetxt(
    f"chi_real{suffix}.txt",
    data_r,
    fmt="%.5f",
    header="T chi_r_integral chi_r_integral_filtered",
    delimiter=" ",
)

data_c = np.column_stack((
    T_values,
    chi_c["integral"],
    chi_c["integral filtered"],
))
np.savetxt(
    f"chi_imag{suffix}.txt",
    data_c,
    fmt="%.5f",
    header="T chi_imag_integral chi_imag_integral_filtered",
    delimiter=" ",
)
