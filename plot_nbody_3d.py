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

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D N-Body Simulation Trajectories')

# Set axis limits (adjust based on your data)
ax.set_xlim(-1.5e11, 1.5e11)
ax.set_ylim(-1.5e11, 1.5e11)
ax.set_zlim(-1.5e11, 1.5e11)

# Plot trajectories and points
lines = [ax.plot([], [], [], 'o-', label=f'Body {i+1}')[0] for i in range(num_bodies)]
points = [ax.plot([], [], [], 'o', ms=10)[0] for i in range(num_bodies)]
ax.legend()

# Initialization function
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    return lines + points

# Animation update function
def update(frame):
    for i in range(num_bodies):
        lines[i].set_data(x_data[i][:frame+1], y_data[i][:frame+1])
        lines[i].set_3d_properties(z_data[i][:frame+1])
        points[i].set_data(x_data[i][frame], y_data[i][frame])
        points[i].set_3d_properties(z_data[i][frame])
    return lines + points

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=False, interval=50)

plt.show()