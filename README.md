# AC Susceptibility of Nanoparticle Assemblies (Micromagnetic)

Micromagnetic simulation code to compute the **real and imaginary parts of the
AC susceptibility** of assemblies of magnetic nanoparticles as a function of
temperature.

Each simulation cell represents a nanoparticle with:
- **Uniaxial anisotropy** with a random easy axis per cell
- **Dipolar interactions** between cells (optional, controlled by the packing fraction)
- An **external AC magnetic field**

Different **container shapes** are implemented (sphere, cube, in-plane disc,
out-of-plane disc, and spheroid variants), and the **packing fraction** can be
set to `1`, `1/2`, `1/8`, or `0` (the latter disables dipolar interactions).

The AC susceptibility is obtained from the simulated time-dependent
magnetization via direct time integration over one field cycle, with an
optional FFT band-pass filter around the excitation frequency.

MoisesZar
moisesg.zarzoza@uclm.es

## Requirements

- Python 3.9+
- [Ubermag](https://ubermag.github.io/) (`oommfc`, `discretisedfield`, `micromagneticmodel`, `micromagneticdata`)
- OOMMF installed and available in `PATH` (Ubermag backend)
- `numpy`, `scipy`, `matplotlib`, `tqdm`, `psutil`

```bash
pip install oommfc discretisedfield micromagneticmodel micromagneticdata numpy scipy matplotlib tqdm psutil
