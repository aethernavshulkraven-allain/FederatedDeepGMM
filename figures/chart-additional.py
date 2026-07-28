import matplotlib.pyplot as plt
import numpy as np

# Use LaTeX for all text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Scenarios and LaTeX labels
tex_scenarios = [r"FEMNIST$_x$", r"FEMNIST$_{x,z}$", r"FEMNIST$_z$", 
                 r"CIFAR10$_x$", r"CIFAR10$_{x,z}$", r"CIFAR10$_z$"]

# Data from the provided LaTeX table
# Structure: [FGMM-GDA (0.1), FGMM-SGDA (0.1), FGMM-GDA (1.0), FGMM-SGDA (1.0)]
means_data = [
    [0.27, 0.23, 0.17, 0.19], # FEMNISTx
    [0.21, 0.24, 0.16, 0.18], # FEMNISTx,z
    [0.29, 0.25, 0.20, 0.23], # FEMNISTz
    [0.26, 0.27, 0.18, 0.15], # CIFAR10x
    [0.29, 0.30, 0.21, 0.13], # CIFAR10x,z
    [1.73, 0.67, 0.37, 0.35]  # CIFAR10z
]

stds_data = [
    [0.04, 0.02, 0.01, 0.03],
    [0.01, 0.04, 0.03, 0.02],
    [0.02, 0.03, 0.04, 0.01],
    [0.01, 0.01, 0.01, 0.02],
    [0.02, 0.01, 0.02, 0.01],
    [0.01, 0.02, 0.05, 0.02]
]

# Group labels for the X-axis
labels = [r"FGDA ($\alpha=0.1$)", r"FGSGDA ($\alpha=0.1$)", r"FGDA ($\alpha=1.0$)", r"FGSGDA ($\alpha=1.0$)"]
# Colors to distinguish between alpha=0.1 and alpha=1.0
colors = ['#1f77b4', '#6baed6', '#ff7f0e', '#ffbb78'] 

# Set figure size to 7 x 1.75 inches
fig, axes = plt.subplots(1, 6, figsize=(7, 2.25))
x = np.arange(len(labels))

for i, ax in enumerate(axes):
    # Plotting grouped bars
    ax.bar(x, means_data[i], yerr=stds_data[i], capsize=1.2, 
           color=colors, alpha=0.8, edgecolor='black', linewidth=0.4)
    
    # Title and axis formatting
    ax.set_title(tex_scenarios[i], fontsize=8, pad=3)
    if i == 0:
        ax.set_ylabel(r"Test MSE", fontsize=8)
    
    # Adjust x-ticks for the 4-bar group
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.tick_params(axis='y', labelsize=6.5, pad=1)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.4)

# Tighten layout to ensure everything fits the 1.75" height
plt.tight_layout(pad=0.2, w_pad=0.4)

# Save as vector PDF
plt.savefig('dirichlet_comparison.pdf', format='pdf', bbox_inches='tight')
# plt.show()