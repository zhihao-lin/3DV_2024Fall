import os
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
from scipy.optimize import least_squares

# Argument parser
parser = argparse.ArgumentParser(description="Warp field estimation for point cloud data.")
parser.add_argument('--pattern', type=str, required=True,
                    help="Specify the data pattern used in data generation (e.g., 'crossing_lines', 'expanding_circle').")
parser.add_argument('--regularization', type=float, default=1.0)
args = parser.parse_args()

def interpolate_transformations(point, control_node_positions, control_node_params, num_neighbors=4):
    """
    Interpolate transformations at a given point from control nodes.
    """
    # Compute distances to control nodes
    distances = np.linalg.norm(control_node_positions - point, axis=1)
    # Find indices of nearest control nodes
    neighbor_indices = np.argsort(distances)[:num_neighbors]
    neighbor_positions = control_node_positions[neighbor_indices]
    neighbor_params = control_node_params[neighbor_indices]  # [theta, tx, ty]

    # Compute weights (inverse distance weighting)
    eps = 1e-8  # Small constant to avoid division by zero
    weights = 1 / (distances[neighbor_indices] + eps)
    weights /= np.sum(weights)

    # Interpolate transformations
    theta = np.dot(weights, neighbor_params[:, 0])
    tx = np.dot(weights, neighbor_params[:, 1])
    ty = np.dot(weights, neighbor_params[:, 2])

    return theta, tx, ty

def compute_data_term(control_node_params, points_t, observed_displacements, control_node_positions):
    num_points = points_t.shape[0]
    residuals = np.zeros((num_points, 2))
    for i in range(num_points):
        point = points_t[i]
        displacement_observed = observed_displacements[i]

        # Interpolate transformations at point
        theta, tx, ty = interpolate_transformations(point, control_node_positions, control_node_params)

        # Apply interpolated transformation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        transformed_point = rotation_matrix @ point + np.array([tx, ty])
        predicted_displacement = transformed_point - point

        # Residual is the difference between predicted and observed displacement
        residuals[i] = predicted_displacement - displacement_observed

    return residuals.flatten()  # Shape: [2 * num_points]

def compute_regularization_term(control_node_params, control_node_positions, edges):
    residuals = []
    for (i, j) in edges:
        # Get node parameters
        theta_i, tx_i, ty_i = control_node_params[i]
        theta_j, tx_j, ty_j = control_node_params[j]

        # Get node positions
        vi = control_node_positions[i]
        vj = control_node_positions[j]

        # Compute rotation matrix at node i
        cos_theta_i = np.cos(theta_i)
        sin_theta_i = np.sin(theta_i)
        Ri = np.array([[cos_theta_i, -sin_theta_i], [sin_theta_i, cos_theta_i]])
        
        # Compute rotation matrix at node j
        cos_theta_j = np.cos(theta_j)
        sin_theta_j = np.sin(theta_j)
        Rj = np.array([[cos_theta_j, -sin_theta_j], [sin_theta_j, cos_theta_j]])

        # Apply transformations to the shared point (vj)
        transformed_vj_i = Ri @ vj + np.array([tx_i, ty_i])  # Transformed by node i
        transformed_vj_j = Rj @ vj + np.array([tx_j, ty_j])  # Transformed by node j

        # Residual: Difference in transformed positions of vj
        residual = transformed_vj_i - transformed_vj_j
        residuals.append(residual)

    return np.concatenate(residuals)  # Shape: [2 * num_edges]

def total_residuals(control_node_params_flat, points_t, observed_displacements, control_node_positions, edges, lambda_reg):
    num_nodes = control_node_positions.shape[0]
    # Reshape control_node_params_flat back to (num_nodes, 3)
    control_node_params = control_node_params_flat.reshape((num_nodes, 3))

    # Compute data term residuals
    data_residuals = compute_data_term(control_node_params, points_t, observed_displacements, control_node_positions)

    # Compute regularization term residuals
    reg_residuals = compute_regularization_term(control_node_params, control_node_positions, edges)

    # Total residuals
    total_residuals = np.concatenate([data_residuals, lambda_reg * reg_residuals])
    return total_residuals

def warp_points(points, control_node_positions, control_node_params, num_neighbors=4, inverse=False):
    warped_points = np.zeros_like(points)
    for i in range(points.shape[0]):
        point = points[i]

        # Interpolate transformations at point
        theta, tx, ty = interpolate_transformations(point, control_node_positions, control_node_params, num_neighbors)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        if inverse:
            # Inverse rotation
            rotation_matrix_inv = rotation_matrix.T  # Transpose of rotation matrix is its inverse
            # Inverse translation
            t = np.array([tx, ty])
            t_inv = -rotation_matrix_inv @ t
            # Apply inverse transformation
            transformed_point = rotation_matrix_inv @ point + t_inv
        else:
            # Apply forward transformation
            transformed_point = rotation_matrix @ point + np.array([tx, ty])

        warped_points[i] = transformed_point

    return warped_points

# Create directories if they don't exist
if os.path.exists('output/warp_fields'):
    shutil.rmtree('output/warp_fields')
os.makedirs('output/warp_fields')
if os.path.exists('output/warped_points'):
    shutil.rmtree('output/warped_points')
os.makedirs('output/warped_points')
if os.path.exists('output/forward_warped_points'):
    shutil.rmtree('output/forward_warped_points')
os.makedirs('output/forward_warped_points')

# Load data
data_array = np.load(f'data/points_{args.pattern}.npy')  # Shape: [num_frames, num_points, 5]
num_frames, num_points, _ = data_array.shape
max_val = np.max(np.abs(data_array[:, :, :2]))

# Assuming the domain spans from x_min to x_max and y_min to y_max
x_min, x_max = -max_val, max_val
y_min, y_max = -max_val, max_val

# Control node grid resolution
grid_size = max_val / 4

# Generate grid of control nodes
x_nodes = np.arange(x_min, x_max + grid_size, grid_size)
y_nodes = np.arange(y_min, y_max + grid_size, grid_size)
X_nodes, Y_nodes = np.meshgrid(x_nodes, y_nodes)
control_node_positions = np.column_stack((X_nodes.ravel(), Y_nodes.ravel()))  # Shape: [num_nodes, 2]

num_nodes = control_node_positions.shape[0]

# Initialize transformations (theta, tx, ty) for each node
# Start with identity transformations
control_node_params = np.zeros((num_nodes, 3))  # [theta, tx, ty]

# Use Delaunay triangulation to find neighboring nodes (outside the loop)
tri = Delaunay(control_node_positions)
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
        edges.add(edge)
edges = list(edges)
num_edges = len(edges)
# visualize the control nodes and edges
plt.figure(figsize=(6, 6))
plt.scatter(control_node_positions[:, 0], control_node_positions[:, 1], c='blue', label='Control Nodes')
for (i, j) in edges:
    plt.plot(
        [control_node_positions[i, 0], control_node_positions[j, 0]],
        [control_node_positions[i, 1], control_node_positions[j, 1]],
        'k--', alpha=0.5
    )
plt.title('Control Nodes and Edges')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axis('off')
plt.tight_layout()
plt.savefig('output/control_nodes.png')
plt.close()

# Initialize accumulated points and colors
accumulated_points = []
accumulated_rgb = []

# Initialize cumulative transformations for control nodes
cumulative_control_node_params = np.zeros_like(control_node_params)  # [theta, tx, ty]

# For each frame t
for t in range(num_frames - 1):
    print(f'Processing frame {t}...')
    frame_t = data_array[t]
    frame_t1 = data_array[t + 1]

    points_t = frame_t[:, :2]
    rgb_t = frame_t[:, 2:]

    points_t1 = frame_t1[:, :2]
    rgb_t1 = frame_t1[:, 2:]

    # Build KD-tree for RGB values in frame t+1
    tree = KDTree(rgb_t1)
    distances, indices = tree.query(rgb_t, k=1)
    matched_indices_t1 = indices.flatten()
    matched_points_t1 = points_t1[matched_indices_t1]

    # Compute initial displacements (observed data)
    observed_displacements = matched_points_t1 - points_t  # Shape: [num_points, 2]

    # Initial parameters (flattened)
    control_node_params_flat = control_node_params.flatten()

    # Regularization weight
    lambda_reg = args.regularization

    # Optimization
    result = least_squares(
        total_residuals,
        control_node_params_flat,
        args=(points_t, observed_displacements, control_node_positions, edges, lambda_reg),
        verbose=2,
        method='lm',  # Levenberg-Marquardt
    )

    # Update control node parameters
    frame_control_node_params = result.x.reshape((num_nodes, 3))

    # Accumulate transformations
    cumulative_control_node_params[:, 0] += frame_control_node_params[:, 0]  # Cumulative rotation
    cumulative_control_node_params[:, 1] += frame_control_node_params[:, 1]  # Cumulative translation x
    cumulative_control_node_params[:, 2] += frame_control_node_params[:, 2]  # Cumulative translation y

    # Warp points from frame t+1 into the canonical frame
    warped_points_t1 = warp_points(points_t1, control_node_positions, cumulative_control_node_params, inverse=True)
    warped_rgb_t1 = rgb_t1  # RGB values remain the same

    # Accumulate points
    if t == 0:
        # Include points from frame 0 (already in canonical frame)
        accumulated_points.append(points_t)
        accumulated_rgb.append(rgb_t)
    accumulated_points.append(warped_points_t1)
    accumulated_rgb.append(warped_rgb_t1)

    # Visualize the warp field
    plt.figure(figsize=(6, 6))
    plt.scatter(control_node_positions[:, 0], control_node_positions[:, 1], c='blue', label='Control Nodes')

    # Plot transformed control nodes
    transformed_nodes = warp_points(control_node_positions, control_node_positions, cumulative_control_node_params)
    plt.scatter(transformed_nodes[:, 0], transformed_nodes[:, 1], c='red', label='Transformed Nodes')

    # Draw lines between original and transformed nodes
    for i in range(num_nodes):
        plt.plot(
            [control_node_positions[i, 0], transformed_nodes[i, 0]],
            [control_node_positions[i, 1], transformed_nodes[i, 1]],
            'k--', alpha=0.5
        )

    plt.title(f'Cumulative Warp Field at Frame {t}')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('off')
    plt.tight_layout()
    # Save the warp field visualization
    plt.savefig(f'output/warp_fields/{t:04d}.png')
    plt.close()

    accumulated_points_array = np.vstack(accumulated_points)
    accumulated_rgb_array = np.vstack(accumulated_rgb)

    # Visualize the accumulated point cloud
    plt.figure(figsize=(6, 6))
    plt.scatter(
        accumulated_points_array[:, 0],
        accumulated_points_array[:, 1],
        c=accumulated_rgb_array,
        s=5
    )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('off')
    plt.title('Accumulated Point Cloud in Canonical Frame')
    plt.tight_layout()
    plt.savefig(f'output/warped_points/{t:04d}.png')
    plt.close()
    
    # Forward warp accumulated points to the live frame (frame t)
    forward_warped_points = warp_points(accumulated_points_array, control_node_positions, cumulative_control_node_params, inverse=False)

    # Visualize forward warped accumulated points
    plt.figure(figsize=(6, 6))
    plt.scatter(
        forward_warped_points[:, 0],
        forward_warped_points[:, 1],
        c=accumulated_rgb_array,
        s=5
    )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('off')
    plt.title(f'Forward Warped Points to Frame {t}')
    plt.tight_layout()
    plt.savefig(f'output/forward_warped_points/{t:04d}.png')
    plt.close()

# Save the visualization as a GIF
import imageio
images = []
for i in range(num_frames - 1):
    warp_field_filename = f'output/warp_fields/{i:04d}.png'
    images.append(imageio.imread(warp_field_filename))
imageio.mimsave('output/warp_fields.gif', images, duration=0.1, loop=0)

images = []
for i in range(num_frames - 1):
    warped_points_filename = f'output/warped_points/{i:04d}.png'
    images.append(imageio.imread(warped_points_filename))
imageio.mimsave('output/warped_points.gif', images, duration=0.1, loop=0)

images = []
for i in range(num_frames - 1):
    forward_warped_points_filename = f'output/forward_warped_points/{i:04d}.png'
    images.append(imageio.imread(forward_warped_points_filename))
imageio.mimsave('output/forward_warped_points.gif', images, duration=0.1, loop=0)