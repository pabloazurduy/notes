# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

###
### run with: `uv run gradient.py``
###


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the constraint function g(x, y, z) = constant
# We'll use:  g(x, y, z) = x^2 + y^2 + z^2 = 4 (a sphere of radius 2)
def g(x, y, z):
    return x**2 + y**2 + z**2

# Gradient of g
def grad_g(x, y, z):
    return np.array([2*x, 2*y, 2*z])

# Choose a point on the surface
point = np.array([1.0, 1.0, np.sqrt(2)])  # This point is on the sphere x^2 + y^2 + z^2 = 4

# Calculate gradient at this point
gradient = grad_g(*point)

# Create the figure
fig = plt.figure(figsize=(14, 6))

# ========== Plot 1: Sphere with gradient vector ==========
ax1 = fig.add_subplot(121, projection='3d')

# Create sphere surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = 2 * np.outer(np.cos(u), np.sin(v))
y_sphere = 2 * np.outer(np.sin(u), np.sin(v))
z_sphere = 2 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere
ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='cyan', edgecolor='none')

# Plot the point
ax1.scatter(*point, color='red', s=100, label=f'Point ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.2f})')

# Plot the gradient vector
ax1.quiver(point[0], point[1], point[2], 
           gradient[0], gradient[1], gradient[2],
           color='red', arrow_length_ratio=0.3, linewidth=3,
           label=f'∇g = ({gradient[0]:.1f}, {gradient[1]:.1f}, {gradient[2]:.2f})')

# Create tangent plane at the point
# The tangent plane equation:  gradient · (r - point) = 0
# Or: 2x(X-x₀) + 2y(Y-y₀) + 2z(Z-z₀) = 0
# Solving for Z: Z = z₀ - (2x/2z)(X-x₀) - (2y/2z)(Y-y₀)

size = 1.5
xx, yy = np.meshgrid(np.linspace(point[0]-size, point[0]+size, 10),
                     np.linspace(point[1]-size, point[1]+size, 10))

# Tangent plane:  gradient · (r - point) = 0
# 2x(X-x₀) + 2y(Y-y₀) + 2z(Z-z₀) = 0
# Z = z₀ - (2x(X-x₀) + 2y(Y-y₀))/(2z)
zz = point[2] - (gradient[0]*(xx-point[0]) + gradient[1]*(yy-point[1]))/gradient[2]

# Plot tangent plane
ax1.plot_surface(xx, yy, zz, alpha=0.5, color='yellow', edgecolor='black', linewidth=0.5)

# Plot some tangent vectors on the plane
tangent_vec1 = np.array([1, 0, -gradient[0]/gradient[2]])  # One tangent direction
tangent_vec2 = np.array([0, 1, -gradient[1]/gradient[2]])  # Another tangent direction

# Normalize for visualization
tangent_vec1 = tangent_vec1 / np.linalg.norm(tangent_vec1) * 1.5
tangent_vec2 = tangent_vec2 / np.linalg.norm(tangent_vec2) * 1.5

ax1.quiver(point[0], point[1], point[2],
           tangent_vec1[0], tangent_vec1[1], tangent_vec1[2],
           color='green', arrow_length_ratio=0.3, linewidth=2,
           label='Tangent vector 1')

ax1.quiver(point[0], point[1], point[2],
           tangent_vec2[0], tangent_vec2[1], tangent_vec2[2],
           color='blue', arrow_length_ratio=0.3, linewidth=2,
           label='Tangent vector 2')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Surface with Gradient and Tangent Plane')
ax1.legend()
ax1.set_box_aspect([1,1,1])

# ========== Plot 2: Different function - Ellipsoid ==========
ax2 = fig.add_subplot(122, projection='3d')

# Define ellipsoid:  g(x, y, z) = x^2/4 + y^2/9 + z^2 = 1
def g2(x, y, z):
    return x**2/4 + y**2/9 + z**2

def grad_g2(x, y, z):
    return np.array([x/2, 2*y/9, 2*z])

# Point on ellipsoid
point2 = np.array([1.5, 2.0, 0.5])

# Calculate gradient at this point
gradient2 = grad_g2(*point2)

# Create ellipsoid surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_ellipsoid = 2 * np.outer(np.cos(u), np.sin(v))
y_ellipsoid = 3 * np.outer(np.sin(u), np.sin(v))
z_ellipsoid = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the ellipsoid
ax2.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                 alpha=0.3, color='lightblue', edgecolor='none')

# Plot the point
ax2.scatter(*point2, color='red', s=100, 
            label=f'Point ({point2[0]:.1f}, {point2[1]:.1f}, {point2[2]:.1f})')

# Plot the gradient vector
ax2.quiver(point2[0], point2[1], point2[2],
           gradient2[0], gradient2[1], gradient2[2],
           color='red', arrow_length_ratio=0.3, linewidth=3,
           label=f'∇g')

# Create tangent plane
size = 1.0
xx2, yy2 = np.meshgrid(np.linspace(point2[0]-size, point2[0]+size, 10),
                       np.linspace(point2[1]-size, point2[1]+size, 10))

zz2 = point2[2] - (gradient2[0]*(xx2-point2[0]) + gradient2[1]*(yy2-point2[1]))/gradient2[2]

ax2.plot_surface(xx2, yy2, zz2, alpha=0.5, color='yellow', edgecolor='black', linewidth=0.5)

# Plot tangent vectors
tangent_vec1_2 = np.array([1, 0, -gradient2[0]/gradient2[2]])
tangent_vec2_2 = np.array([0, 1, -gradient2[1]/gradient2[2]])

tangent_vec1_2 = tangent_vec1_2 / np.linalg.norm(tangent_vec1_2) * 1.5
tangent_vec2_2 = tangent_vec2_2 / np.linalg.norm(tangent_vec2_2) * 1.5

ax2.quiver(point2[0], point2[1], point2[2],
           tangent_vec1_2[0], tangent_vec1_2[1], tangent_vec1_2[2],
           color='green', arrow_length_ratio=0.3, linewidth=2,
           label='Tangent vector 1')

ax2.quiver(point2[0], point2[1], point2[2],
           tangent_vec2_2[0], tangent_vec2_2[1], tangent_vec2_2[2],
           color='blue', arrow_length_ratio=0.3, linewidth=2,
           label='Tangent vector 2')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Ellipsoid with Gradient and Tangent Plane')
ax2.legend()
ax2.set_box_aspect([1,1,1])

plt.tight_layout()
plt.show()

# ========== Verify orthogonality numerically ==========
print("=" * 60)
print("ORTHOGONALITY VERIFICATION")
print("=" * 60)

print("\n--- Sphere Example ---")
print(f"Point: {point}")
print(f"Gradient: {gradient}")
print(f"Tangent vector 1: {tangent_vec1}")
print(f"Tangent vector 2: {tangent_vec2}")

dot1 = np.dot(gradient, tangent_vec1)
dot2 = np.dot(gradient, tangent_vec2)

print(f"\nGradient · Tangent1 = {dot1:.10f}  (should be ≈ 0)")
print(f"Gradient · Tangent2 = {dot2:.10f}  (should be ≈ 0)")

print("\n--- Ellipsoid Example ---")
print(f"Point: {point2}")
print(f"Gradient: {gradient2}")
print(f"Tangent vector 1: {tangent_vec1_2}")
print(f"Tangent vector 2: {tangent_vec2_2}")

dot1_2 = np.dot(gradient2, tangent_vec1_2)
dot2_2 = np.dot(gradient2, tangent_vec2_2)

print(f"\nGradient · Tangent1 = {dot1_2:.10f}  (should be ≈ 0)")
print(f"Gradient · Tangent2 = {dot2_2:.10f}  (should be ≈ 0)")

print("\n" + "=" * 60)
print("The dot products are approximately zero, confirming")
print("that the gradient is perpendicular to the tangent plane!")
print("=" * 60)