import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Read positions from file
def read_positions(filename):
    timesteps = []
    positions = []
    current_timestep = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Time step"):
                if current_timestep:
                    positions.append(current_timestep)
                    current_timestep = []
            elif line:
                x, y, z = map(float, line.split())
                current_timestep.append((x, y, z))
        if current_timestep:
            positions.append(current_timestep)
    return positions

# Load data
positions = read_positions('positions.txt')
num_bodies = len(positions[0])
num_steps = len(positions)

# Prepare data for plotting
x_data = [[] for _ in range(num_bodies)]
y_data = [[] for _ in range(num_bodies)]
z_data = [[] for _ in range(num_bodies)]
for t in range(num_steps):
    for i in range(num_bodies):
        x_data[i].append(positions[t][i][0])
        y_data[i].append(positions[t][i][1])
        z_data[i].append(positions[t][i][2])

# Calculate center of mass trajectory
cm_x = np.mean(x_data, axis=0)
cm_y = np.mean(y_data, axis=0)
cm_z = np.mean(z_data, axis=0)

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')

# Dynamically set axis limits based on data
padding = 0.2  # 20% padding around data
x_range = max(max(np.max(x_data) - np.min(x_data), np.max(cm_x) - np.min(cm_x)), 1e10)
y_range = max(max(np.max(y_data) - np.min(y_data), np.max(cm_y) - np.min(cm_y)), 1e10)
z_range = max(max(np.max(z_data) - np.min(z_data), np.max(cm_z) - np.min(cm_z)), 1e10)
x_mid = (np.max(x_data) + np.min(x_data)) / 2
y_mid = (np.max(y_data) + np.min(y_data)) / 2
z_mid = (np.max(z_data) + np.min(z_data)) / 2
ax.set_xlim(x_mid - x_range * (1 + padding) / 2, x_mid + x_range * (1 + padding) / 2)
ax.set_ylim(y_mid - y_range * (1 + padding) / 2, y_mid + y_range * (1 + padding) / 2)
ax.set_zlim(z_mid - z_range * (1 + padding) / 2, z_mid + z_range * (1 + padding) / 2)

# Plot trajectories and points with distinct colors
colors = ['b', 'r', 'g', 'm', 'c']  # Blue, Red, Green, Magenta, Cyan
lines = [ax.plot([], [], [], 'o-', label=f'Body {i+1}', color=colors[i % len(colors)])[0] for i in range(num_bodies)]
points = [ax.plot([], [], [], 'o', ms=10, color=colors[i % len(colors)])[0] for i in range(num_bodies)]
cm_line, = ax.plot([], [], [], 'k--', label='Center of Mass')  # Dashed black line
cm_point, = ax.plot([], [], [], 'ko', ms=8)  # Black dot
ax.legend()

# Add title with time step
title = ax.set_title('3D N-Body Simulation: Time Step 0')

# Initialization function
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    cm_line.set_data([], [])
    cm_line.set_3d_properties([])
    cm_point.set_data([], [])
    cm_point.set_3d_properties([])
    title.set_text('3D N-Body Simulation: Time Step 0')
    return lines + points + [cm_line, cm_point, title]

# Animation update function
def update(frame):
    for i in range(num_bodies):
        lines[i].set_data(x_data[i][:frame+1], y_data[i][:frame+1])
        lines[i].set_3d_properties(z_data[i][:frame+1])
        points[i].set_data(x_data[i][frame], y_data[i][frame])
        points[i].set_3d_properties(z_data[i][frame])
    cm_line.set_data(cm_x[:frame+1], cm_y[:frame+1])
    cm_line.set_3d_properties(cm_z[:frame+1])
    cm_point.set_data(cm_x[frame], cm_y[frame])
    cm_point.set_3d_properties(cm_z[frame])
    title.set_text(f'3D N-Body Simulation: Time Step {frame+1}')
    return lines + points + [cm_line, cm_point, title]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=False, interval=50)

# Display the plot
plt.show()
