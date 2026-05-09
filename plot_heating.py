import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
# Matplotlib Plot Formatting
font = {'weight' : 'normal', 'size'   : 18}
plt.rc('font', **font)

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm


# ── 1) Your original 10×10 data ─────────────────────────────────────────────
n_grid = np.arange(10)                   # discrete X coords (0,1,…,9)
gamma_grid = np.arange(10)               # discrete Y coords (0,1,…,9)
timed_fidelities = np.random.rand(10, 10)  # replace with your 10×10 Z array

data = np.load('data/heating_grid_data.npz')

# Access the arrays stored in the file
n_grid = data['n_grid']#[:11,:]
gamma_grid = data['kappa_grid']#[:11,:]
timed_fidelities_grid = data['timed_fidelities_grid']#[:11,:]
ns = data['ns']#[:11]
gammas = data['kappas']

print(ns.shape)
print(gammas.shape)
print(timed_fidelities_grid.shape)
print(timed_fidelities_grid)
# ── 2) Interpolate ONLY in Y ────────────────────────────────────────────────
# target 100 points (for smooth vertical variation)
y_new = np.linspace(gammas[0], gammas[-1], 300)

# build per‑column interpolator
interp_f = interp1d(gammas, timed_fidelities_grid.T, axis=0, kind='linear')
Z_interp = interp_f(y_new)    # now shape (100, 10)

# ── 3) Build edge arrays for pcolormesh (shading='flat') ───────────────────
# X edges: one block per integer, centered on n_grid
x_edges = np.concatenate([ns - 0.5, [ns[-1] + 0.5]])   # shape (11,)

# Y edges: half‑step beyond first/last of y_new
dy = y_new[1] - y_new[0]
y_edges = np.concatenate([y_new - dy/2, [y_new[-1] + dy/2]])   # shape (101,)

# make meshgrid of edges
X_e, Y_e = np.meshgrid(x_edges, y_edges)   # both shape (101, 11)

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np



import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Return a truncated version of a colormap"""
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Example: remove the lightest 20% of viridis
viridis = cm.get_cmap("viridis")
viridis_trunc = truncate_colormap(viridis, 0.0, 0.9)  # only first 80%


# Define color map: green to yellow
colors = [
    (1.0, 1.0, 1.0),  # yellow (#ffff00)
    (0.0, 0.6, 0.3),  # green (#00ff00)
]

green_yellow = LinearSegmentedColormap.from_list("green_yellow", colors)

# ── 4) Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
pcm = ax.pcolormesh(
    X_e, Y_e, Z_interp,
    cmap="RdYlGn",      #   RdYlGn
    shading='flat',
    vmin=np.min(timed_fidelities_grid.reshape(-1)[timed_fidelities_grid.reshape(-1)!= 0 ]),  
    vmax=1     # and your desired max
)

# Overlay zero values in white
zero_mask = (Z_interp < np.min(timed_fidelities_grid.reshape(-1)[timed_fidelities_grid.reshape(-1)!= 0 ]))
if np.any(zero_mask):
    ax.pcolormesh(
        X_e, Y_e, np.where(zero_mask, 1, np.nan),  # dummy values just to show color
        cmap=ListedColormap(['white']),
        shading='flat'
    )

# center xticks on each discrete X
ax.set_xticks(ns)
ax.set_xticklabels(ns)

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Heating Rate (in units of $\Omega_t$)')

fig.colorbar(pcm, ax=ax, label='Fidelity')
plt.tight_layout()
fig.subplots_adjust(left=0.12, right=1., top=0.98, bottom=0.12)

plt.savefig("figures/heating_plot.png")
plt.show()
