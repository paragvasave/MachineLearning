import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Coordinates of the two points
point1 = (5.1, 3.5, 1.4, 0.2)
point2 = (6.4, 2.9, 4.3, 1.3)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z coordinates of each point
x1, y1, z1, _ = point1
x2, y2, z2, _ = point2

# Plot the points
ax.scatter(x1, y1, z1, c='red', marker='o', label='Point 1')
ax.scatter(x2, y2, z2, c='blue', marker='o', label='Point 2')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a legend
ax.legend()

# Show the plot
plt.show()