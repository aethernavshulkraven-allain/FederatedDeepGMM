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
short_models = ["GDA", "OAdam", "FGDA", "SGDA", "FSGDA"]

# Data
means_data = [
    [1.11, 0.50, 0.21, 0.40, 0.19], [0.46, 0.24, 0.19, 0.14, 0.20], 
    [0.42, 0.10, 0.24, 0.11, 0.23], [0.19, 0.55, 0.25, 0.20, 0.22], 
    [0.24, 0.40, 0.24, 0.19, 0.22], [0.13, 0.13, 1.70, 0.24, 0.52]
]

stds_data = [
    [0.01, 0.00, 0.02, 0.01, 0.01], [0.09, 0.00, 0.03, 0.02, 0.00], 
    [0.01, 0.00, 0.01, 0.02, 0.01], [0.01, 0.10, 0.03, 0.08, 0.08], 
    [0.00, 0.05, 0.03, 0.03, 0.02], [0.01, 0.03, 0.26, 0.01, 0.06]
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Set figure size to 7 x 1.75 inches
fig, axes = plt.subplots(1, 6, figsize=(7, 1.25))
x = np.arange(len(short_models))

for i, ax in enumerate(axes):
    ax.bar(x, means_data[i], yerr=stds_data[i], capsize=1.5, 
           color=colors, alpha=0.8, edgecolor='black', linewidth=0.4)
    
    # Tightened styling for smaller height
    ax.set_title(tex_scenarios[i], fontsize=8, pad=3)
    if i == 0:
        ax.set_ylabel(r"Test MSE", fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(short_models, rotation=90, fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5, pad=1)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.4)

# Use tight_layout with small padding to maximize bar area
plt.tight_layout(pad=0.2, w_pad=0.5)

# Save as vector PDF
plt.savefig('scenarios_short.pdf', format='pdf', bbox_inches='tight')
# plt.show()