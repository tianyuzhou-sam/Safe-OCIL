import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Create main figure with two subplots
fig = plt.figure(figsize=(9, 9))
ax1 = plt.subplot(1, 1, 1)

# Create inset axes for zoom
# ax2 = plt.axes([0.2, 0.3, 0.3, 0.3])  # [left, bottom, width, height] - center-left

# Generate x values
x = np.linspace(-5, 5, 1000)
# x_zoom = np.linspace(-1, 1, 1000)

# Plot ReLU function in main plot
relu = np.maximum(0, x)
ax1.plot(x, relu, label='ReLU', linestyle='--', color='black', linewidth=5)

# Plot softplus for different beta values in main plot
betas = [1, 1/2, 1/10]
for beta in betas:
    softplus = beta * np.log(1 + np.exp(x/beta))
    ax1.plot(x, softplus, label=f'Softplus β={beta}', linewidth=10)

ax1.plot(x, relu, label='ReLU', linestyle='--', color='black', linewidth=5)

ax1.grid(True)
ax1.legend(['ReLU', 'β=1.0', 'β=0.5', 'β=0.1'], loc='upper left', fontsize=30)
ax1.set_xlabel('x', fontsize=30)
ax1.set_ylabel('$ \\phi_{\\beta}(x) $', fontsize=30)
# ax1.set_title('ReLU vs Softplus Functions', fontsize=30)
ax1.tick_params(labelsize=30)

# Create zoomed plot
# relu_zoom = np.maximum(0, x_zoom)
# ax2.plot(x_zoom, relu_zoom, linestyle='--', color='black', linewidth=3)

# for beta in betas:
#     softplus_zoom = beta * np.log(1 + np.exp(x_zoom/beta))
#     ax2.plot(x_zoom, softplus_zoom, linewidth=3)

# ax2.grid(True)
# ax2.set_xlim(-0.5, 0.5)
# ax2.set_ylim(0, 1)
# ax2.tick_params(labelsize=12)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-0.2, 3.8)
ax1.yaxis.set_major_locator(MultipleLocator(1))
plt.tight_layout()


fig = plt.figure(figsize=(9, 9))
ax2 = plt.subplot(1, 1, 1)
# Second plot: Barrier function with beta fixed at 0.1, alpha varying
beta_fixed = 0.1
alphas = [0.1, 0.5, 1]
ax2.plot(x, relu, label='ReLU', linestyle='--', color='black', linewidth=5)
for alpha in alphas:
    barrier = (beta_fixed / alpha) * np.log(1 + np.exp(x / beta_fixed))
    ax2.plot(x, barrier, label=f'Barrier $\\alpha$={alpha}', linewidth=10)

ax2.plot(x, relu, label='ReLU', linestyle='--', color='black', linewidth=5)


ax2.grid(True)
ax2.legend(['ReLU', '$\\alpha$=0.1', '$\\alpha$=0.5', '$\\alpha$=1.0'], loc='upper left', fontsize=30)
ax2.set_xlabel('x', fontsize=30)
ax2.set_ylabel('$ \\frac{1}{\\alpha} \\phi_{\\beta=0.1}(x) $', fontsize=30)
ax2.tick_params(labelsize=30)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-0.2, 3.8)
ax2.yaxis.set_major_locator(MultipleLocator(1))

plt.tight_layout()
plt.show()
