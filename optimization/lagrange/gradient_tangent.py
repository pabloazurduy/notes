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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ========== PLOT 1: Level curves with gradient and tangent ==========
ax1 = axes[0]

# Create grid
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)

# f(x,y) = x² + y²
F = X**2 + Y**2

# Plot level curves
levels = [1, 2, 4, 6, 8, 10, 12]
contours = ax1.contour(X, Y, F, levels=levels, cmap='Blues', linewidths=2)
ax1.clabel(contours, inline=True, fontsize=10, fmt='f=%g')

# Choose a point on the level curve f = 4 (circle of radius 2)
x0, y0 = 1.2, np.sqrt(4 - 1.2**2)  # Point on circle x² + y² = 4
f_val = x0**2 + y0**2

ax1.scatter([x0], [y0], color='red', s=200, zorder=5, edgecolors='black', linewidths=2)
ax1.text(x0 + 0.2, y0 + 0.3, f'Point ({x0:.1f}, {y0:.2f})\nf = {f_val:.1f}', fontsize=10)

# Calculate gradient at this point:  ∇f = (2x, 2y)
grad_x = 2 * x0
grad_y = 2 * y0

# Normalize for visualization
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_x_norm = grad_x / grad_mag * 1.5
grad_y_norm = grad_y / grad_mag * 1.5

# Plot gradient vector
ax1.quiver(x0, y0, grad_x_norm, grad_y_norm, 
          angles='xy', scale_units='xy', scale=1,
          color='red', width=0.03, label=f'∇f = (2x, 2y) = ({grad_x:.2f}, {grad_y:.2f})')

# Tangent direction (perpendicular to gradient)
# If gradient is (a, b), tangent is (-b, a) or (b, -a)
tangent_x = -grad_y / grad_mag * 1.5
tangent_y = grad_x / grad_mag * 1.5

# Plot tangent vector
ax1.quiver(x0, y0, tangent_x, tangent_y,
          angles='xy', scale_units='xy', scale=1,
          color='green', width=0.03, label='Tangent direction')

# Draw the tangent LINE
t = np.linspace(-2, 2, 100)
tangent_line_x = x0 + t * (-grad_y / grad_mag)
tangent_line_y = y0 + t * (grad_x / grad_mag)
ax1.plot(tangent_line_x, tangent_line_y, 'g--', linewidth=2, alpha=0.7, label='Tangent line')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Level Curves of f(x,y) = x² + y²\nGradient ⊥ Tangent Line', fontsize=13, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)

# Verify perpendicularity
dot_product = grad_x * (-grad_y) + grad_y * (grad_x)  # Should be 0
ax1.text(-3.5, -3, f'∇f · tangent = {dot_product:.6f}\n(= 0, perpendicular!)', 
        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

# ========== PLOT 2: Multiple points showing gradient field ==========
ax2 = axes[1]

# Plot level curves
contours2 = ax2.contour(X, Y, F, levels=levels, cmap='Blues', linewidths=2, alpha=0.7)

# Plot gradient vectors at multiple points on different level curves
for radius in [1, 2, 3]: 
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        px = radius * np.cos(angle)
        py = radius * np.sin(angle)
        
        # Gradient at this point
        gx = 2 * px
        gy = 2 * py
        g_mag = np.sqrt(gx**2 + gy**2)
        
        # Plot gradient (red) - pointing outward
        ax2.quiver(px, py, gx/g_mag*0.5, gy/g_mag*0.5,
                  angles='xy', scale_units='xy', scale=1,
                  color='red', width=0.02, alpha=0.8)
        
        # Plot tangent (green) - pointing along circle
        ax2.quiver(px, py, -gy/g_mag*0.5, gx/g_mag*0.5,
                  angles='xy', scale_units='xy', scale=1,
                  color='green', width=0.02, alpha=0.8)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Gradient Field (red) vs Tangent Field (green)\nGradients point OUTWARD (⊥ to circles)', 
             fontsize=13, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)

# Add legend manually
ax2.plot([], [], 'r-', linewidth=3, label='∇f (perpendicular to level curve)')
ax2.plot([], [], 'g-', linewidth=3, label='Tangent (along level curve)')
ax2.legend(loc='upper right', fontsize=9)

# ========== PLOT 3: 3D surface with tangent plane ==========
ax3 = fig.add_subplot(133, projection='3d')

# Create surface
x_3d = np.linspace(-2, 2, 50)
y_3d = np.linspace(-2, 2, 50)
X3, Y3 = np.meshgrid(x_3d, y_3d)
Z3 = X3**2 + Y3**2

# Plot surface
ax3.plot_surface(X3, Y3, Z3, cmap='viridis', alpha=0.6, edgecolor='none')

# Point on surface
x0_3d, y0_3d = 1.0, 1.0
z0_3d = x0_3d**2 + y0_3d**2

ax3.scatter([x0_3d], [y0_3d], [z0_3d], color='red', s=200, zorder=5)

# Tangent PLANE to the surface at this point
# f(x,y) = x² + y², so the tangent plane is: 
# z - z₀ = ∂f/∂x(x₀,y₀)(x - x₀) + ∂f/∂y(x₀,y₀)(y - y₀)
# z - 2 = 2(1)(x - 1) + 2(1)(y - 1)
# z = 2x + 2y - 2

xx, yy = np.meshgrid(np.linspace(x0_3d - 1, x0_3d + 1, 10),
                     np.linspace(y0_3d - 1, y0_3d + 1, 10))
zz = z0_3d + 2*x0_3d*(xx - x0_3d) + 2*y0_3d*(yy - y0_3d)

ax3.plot_surface(xx, yy, zz, color='yellow', alpha=0.7, edgecolor='black', linewidth=0.5)

# Draw gradient vector in the xy-plane (at z=0 for visibility)
ax3.quiver(x0_3d, y0_3d, 0, 2*x0_3d, 2*y0_3d, 0,
          color='red', arrow_length_ratio=0.3, linewidth=3, label='∇f in xy-plane')

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z = f(x,y)')
ax3.set_title('3D View: Surface f(x,y) = x² + y²\nwith Tangent Plane (yellow)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ========== Print the math ==========
print("=" * 70)
print("CALCULATING GRADIENT AND TANGENT LINE")
print("=" * 70)

print(f"\nFunction: f(x,y) = x² + y²")
print(f"\nPoint: ({x0:.2f}, {y0:.2f})")
print(f"Function value: f({x0:.2f}, {y0:.2f}) = {f_val:.2f}")

print(f"\n" + "-" * 70)
print("GRADIENT:")
print("-" * 70)
print(f"∇f = (∂f/∂x, ∂f/∂y) = (2x, 2y)")
print(f"∇f({x0:.2f}, {y0:.2f}) = ({grad_x:.2f}, {grad_y:.2f})")
print(f"\nThis vector points OUTWARD from the origin,")
print(f"PERPENDICULAR to the level curve (circle).")

print(f"\n" + "-" * 70)
print("TANGENT LINE TO LEVEL CURVE:")
print("-" * 70)
print(f"The level curve is:  x² + y² = {f_val:.2f} (a circle)")
print(f"\nThe tangent line satisfies: ∇f · (x - x₀, y - y₀) = 0")
print(f"  {grad_x:.2f}(x - {x0:.2f}) + {grad_y:.2f}(y - {y0:.2f}) = 0")
print(f"  {grad_x:.2f}x + {grad_y:.2f}y = {grad_x*x0 + grad_y*y0:.2f}")

print(f"\n" + "-" * 70)
print("TANGENT DIRECTION VECTOR:")
print("-" * 70)
print(f"If ∇f = (a, b), then tangent direction = (-b, a) or (b, -a)")
print(f"∇f = ({grad_x:.2f}, {grad_y:.2f})")
print(f"Tangent direction = ({-grad_y:.2f}, {grad_x:.2f})")
print(f"\nVerify perpendicularity:")
print(f"∇f · tangent = ({grad_x:.2f})({-grad_y:.2f}) + ({grad_y:.2f})({grad_x:.2f})")
print(f"            = {grad_x * (-grad_y):.2f} + {grad_y * grad_x:.2f}")
print(f"            = {grad_x * (-grad_y) + grad_y * grad_x:.2f}  ✓ (perpendicular!)")

print(f"\n" + "=" * 70)
print("GENERAL FORMULA")
print("=" * 70)
print("""
For any function f(x,y) and a level curve f(x,y) = c:

1.  GRADIENT at point (x₀, y₀):
   ∇f = (∂f/∂x, ∂f/∂y)

2. TANGENT LINE to level curve at (x₀, y₀):
   ∇f(x₀,y₀) · (x - x₀, y - y₀) = 0
   
   Expanded: 
   ∂f/∂x|₍ₓ₀,ᵧ₀₎ (x - x₀) + ∂f/∂y|₍ₓ₀,ᵧ₀₎ (y - y₀) = 0

3. TANGENT DIRECTION vector:
   If ∇f = (a, b), then tangent direction = (-b, a)
   
4. PERPENDICULARITY: 
   ∇f · tangent = (a, b) · (-b, a) = -ab + ab = 0  ✓
""")
print("=" * 70)