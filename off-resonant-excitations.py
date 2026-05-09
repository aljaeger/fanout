import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

t = np.linspace(0,5,500)

omega_t = 1
omega_c = 7

plt.figure(figsize=(5,3))
for m in np.arange(0, 4):
    z = (1 - 2 * m* omega_t**2 / (omega_c**2 + m * omega_t**2) * np.sin(t/2*np.sqrt(omega_c**2 + m*omega_t**2))**2)**2
    plt.plot(t, z, label=f"$m={m}$")
plt.legend()
plt.ylim(0.5, 1.03)
plt.xlabel("Time (a.u.)")
plt.ylabel("Fidelity")
plt.grid()
plt.tight_layout()
plt.savefig("figures/off-resonant-excitations.pdf")
