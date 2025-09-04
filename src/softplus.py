import numpy as np
import matplotlib.pyplot as plt

# Create main figure
fig = plt.figure(figsize=(10, 6))

# Create main plot
ax1 = plt.gca()

# Create inset axes for zoom
ax2 = plt.axes([0.2, 0.35, 0.3, 0.3])  # [left, bottom, width, height] - center-left

# Generate x values
x = np.linspace(-5, 5, 1000)
x_zoom = np.linspace(-1, 1, 1000)

# Plot ReLU function in main plot
relu = np.maximum(0, x)
ax1.plot(x, relu, label='ReLU', linestyle='--', color='black', linewidth=3)

# Plot softplus for different beta values in main plot
betas = [1, 1/2, 1/10]
for beta in betas:
    softplus = beta * np.log(1 + np.exp(x/beta))
    ax1.plot(x, softplus, label=f'Softplus β={beta}', linewidth=3)

ax1.grid(True)
ax1.legend(loc='upper left', fontsize=14)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('y', fontsize=14)
# ax1.set_title('ReLU vs Softplus Functions', fontsize=14)
ax1.tick_params(labelsize=14)

# Create zoomed plot
relu_zoom = np.maximum(0, x_zoom)
ax2.plot(x_zoom, relu_zoom, linestyle='--', color='black', linewidth=3)

for beta in betas:
    softplus_zoom = beta * np.log(1 + np.exp(x_zoom/beta))
    ax2.plot(x_zoom, softplus_zoom, linewidth=3)

ax2.grid(True)
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(0, 1)
ax2.tick_params(labelsize=12)

plt.show()
