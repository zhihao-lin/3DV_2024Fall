import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import matplotlib.animation as animation
from typing import List, Tuple

def load_frames():

    frames = []
    frame_idx = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    for i in frame_idx:
        frame = cv2.imread("robot/output_{:04d}.png".format(i), cv2.IMREAD_UNCHANGED)
        frame = (frame[:, :, 3] == 255).astype(np.uint8) * 255
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.copyMakeBorder(frame, 128, 128, 128, 128, cv2.BORDER_CONSTANT, value=0)
        frames.append(frame)
        

    return frames

def generate_robot_contour(max_radius=1.2):
    frames = load_frames()

    xs = []
    ys = []

    mx = None
    my = None
    scale = None

    for i in range(0, len(frames)):
        _, binary_image = cv2.threshold(frames[i], 0.5, 1, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x = np.float32(contours[0][::-1, :, 0].flatten())
        y = np.float32(contours[0][::-1, :, 1].flatten())

        if(mx is None):
            mx = np.mean(x)
            my = np.mean(y)
            
        
        x -= mx
        y -= my

        if(scale is None):
            scale = max_radius / np.maximum(np.max(np.abs(x)), np.max(np.abs(y)))


        x *= scale
        y *= -scale

        xs.append(x[::5])
        ys.append(y[::5])
    
    return xs, ys



#
# Camera Trajectory and depth sampling code acquired from Prof. Shenlong Wang's implementation of 2D Kinect Fusion 
#


def generate_camera_trajectory(num_points=20, radius_min=1.5, radius_max=2, oscillations=2, fov=np.pi/2):
    """
    Generate a closed continuous 2D camera trajectory storing location, lookat vector, and FOV.
    
    Parameters:
    - num_points: Number of points in the trajectory.
    - radius_min: Minimum radius from the origin.
    - radius_max: Maximum radius from the origin.
    - oscillations: Number of oscillations in the radial direction.
    - fov: Field of view in radians.
    
    Returns:
    - cameras: A structured array with 'position', 'lookat', and 'fov' for each camera.
    """
    # Generate angles for the camera trajectory and radii for positions
    angles = np.linspace(0, 2 * np.pi, num_points)
    radii = radius_min + (radius_max - radius_min) * 0.5 * (1 + np.sin(oscillations * angles))
    
    # Compute camera positions in Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    positions = np.column_stack((x, y))  # Shape: (num_points, 2)
    
    # Compute look-at vectors (pointing to the origin)
    lookat_vectors = -positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]
    
    # Create the output structured array
    cameras = np.zeros(num_points, dtype=[('position', float, 2), ('lookat', float, 2), ('fov', float)])
    cameras['position'] = positions
    cameras['lookat'] = lookat_vectors
    cameras['fov'] = fov  # Assign the same FOV for all cameras
    
    return cameras


def plot_camera_trajectory_with_FOV(trajectory, origin=(0, 0), FOV = np.pi / 2, sensor_size=0.2):
    """
    Plot the camera trajectory with isosceles triangles representing the camera's field of view.
    
    Parameters:
    - trajectory: A NumPy array of shape (num_points, 2) representing the (x, y) camera positions.
    - origin: The point the camera is looking at (default is (0, 0)).
    - focal_length: The focal length of the camera.
    - sensor_size: The size of the camera's sensor.
    """
    
    # plt.figure(figsize=(6, 6))
    
    # Plot the camera trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Camera Trajectory", linestyle='--', color='gray')
    
    # Plot the origin (the camera's look-at target)
    plt.scatter([origin[0]], [origin[1]], color='red', label="Origin (Camera Target)", zorder=5)
    
    # Plot the camera FOV for each point
    for position in trajectory:
        lookat_vector = np.array(origin) - position  # Look-at vector pointing to the origin
        lookat_direction = lookat_vector / np.linalg.norm(lookat_vector)  # Normalize
        
        # Compute the two sides of the isosceles triangle (camera's FOV)
        half_angle = FOV / 2
        base_length = sensor_size  # The length of the imaging plane's base
        
        # Rotate the lookat vector by ±half_angle to get the two sides of the triangle
        rotation_matrix_left = np.array([[np.cos(half_angle), -np.sin(half_angle)],
                                         [np.sin(half_angle), np.cos(half_angle)]])
        rotation_matrix_right = np.array([[np.cos(-half_angle), -np.sin(-half_angle)],
                                          [np.sin(-half_angle), np.cos(-half_angle)]])
        
        # Endpoints of the triangle's base
        left_side = np.dot(rotation_matrix_left, lookat_direction) * base_length
        right_side = np.dot(rotation_matrix_right, lookat_direction) * base_length
        
        # Plot the triangle representing the camera's FOV with a solid black boundary
        triangle_points = np.array([position, position + left_side, position + right_side])
        plt.fill(triangle_points[:, 0], triangle_points[:, 1], 'b', alpha=0.5, edgecolor='black', lw=1)  # Add black boundary
        
        # Plot a circle at the camera's position (vertex of the triangle)
        plt.scatter(position[0], position[1], color='blue', edgecolor='black', zorder=10)  # Circle dot at the camera location
    
def ray_contour_intersection(camera_pos, ray_direction, contour_x, contour_y):
    """
    Compute the first intersection point between a ray and the heart-shaped contour.

    Parameters:
    - camera_pos: The position of the camera (origin of the rays).
    - ray_direction: A NumPy array of shape (2,) representing the direction of the ray.
    - contour_x, contour_y: The x and y coordinates of the contour points.

    Returns:
    - depth: The distance (depth) to the first intersection. -1 if no intersection.
    """
    intersections = []
    
    # Convert contour to segments (x1, y1) -> (x2, y2)
    for i in range(len(contour_x) - 1):
        p1 = np.array([contour_x[i], contour_y[i]])
        p2 = np.array([contour_x[i + 1], contour_y[i + 1]])
        v1 = camera_pos - p1
        v2 = p2 - p1
        v3 = np.array([-ray_direction[1], ray_direction[0]])  # Perpendicular to ray_direction

        denom = np.dot(v2, v3)
        if abs(denom) < 1e-6:
            continue  # Lines are parallel

        t1 = np.cross(v2, v1) / denom
        t2 = np.dot(v1, v3) / denom

        if 0 <= t2 <= 1 and t1 >= 0:
            intersections.append(t1)  # t1 is the depth along the ray

    if intersections:
        return min(intersections)  # Return the closest intersection
    return -1  # No intersection

def compute_depth_image(camera, contour_x, contour_y, num_rays=128):
    """
    Compute depth image using the camera's location, lookat vector, and FOV.
    
    Parameters:
    - camera: A dictionary containing camera 'position', 'lookat', and 'fov'.
    - contour_x, contour_y: Coordinates of the heart-shaped contour.
    - num_rays: The number of rays to shoot from the camera.
    
    Returns:
    - depths: A NumPy array of shape (num_rays,) with the depth values.
    """
    fov = camera['fov']
    camera_pos = camera['position']
    lookat_vector = camera['lookat']
    
    # Generate ray directions within the FOV
    angles = np.linspace(-fov / 2, fov / 2, num_rays)
    ray_directions = np.column_stack([
        np.cos(angles) * lookat_vector[0] - np.sin(angles) * lookat_vector[1],
        np.sin(angles) * lookat_vector[0] + np.cos(angles) * lookat_vector[1]
    ])
    
    # Initialize an array to store the depths of each ray
    depths = np.full(num_rays, -1.0)  # Default depth = -1 for no intersection
    
    # Loop through each ray direction and calculate its depth
    for i in range(num_rays):
        ray_direction = ray_directions[i]
        depth = ray_contour_intersection(camera_pos, ray_direction, contour_x, contour_y)
        
        # Assign the depth (if valid)
        if depth >= 0:
            depths[i] = depth
    
    return depths


def plot_depth_image(depths):
    """Plot the 128-dimensional 1D depth image."""
    plt.plot(depths, 'b-', marker='o', label="Depth")
    plt.axhline(y=-1, color='r', linestyle='--', label="No Intersection")
    plt.title("1D Depth Image")
    plt.xlabel("Ray Index")
    plt.ylabel("Depth")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rays_with_intersections(camera, ind, contour_x, contour_y, num_rays=16):
    """Plot 16 rays from the camera and their intersections with the contour."""
    camera_pos = camera['position'][ind]
    fov = camera['fov'][ind]
    lookat_vector = camera['lookat'][ind]
    
    angles = np.linspace(-fov / 2, fov / 2, num_rays)
    ray_directions = np.column_stack([
        np.cos(angles) * lookat_vector[0] - np.sin(angles) * lookat_vector[1],
        np.sin(angles) * lookat_vector[0] + np.cos(angles) * lookat_vector[1]
    ])
    
    # Compute depths and plot the rays with intersections
    for ray_direction in ray_directions:
        depth = ray_contour_intersection(camera_pos, ray_direction, contour_x, contour_y)

        # Plot the ray
        end_point = camera_pos + 2 * ray_direction if depth == -1 else camera_pos + depth * ray_direction
        plt.plot([camera_pos[0], end_point[0]], [camera_pos[1], end_point[1]], 'k--')

        # Plot the intersection point
        if depth != -1:
            intersection_point = camera_pos + depth * ray_direction
            plt.plot(intersection_point[0], intersection_point[1], 'go')  # Mark the intersection


def calc_rays_intersections(camera, contour_x, contour_y, num_rays=256):
    camera_pos = camera['position']
    fov = camera['fov']
    lookat_vector = camera['lookat']

    angles = np.linspace(-fov / 2, fov / 2, num_rays)
    ray_directions = np.column_stack([
        np.cos(angles) * lookat_vector[0] - np.sin(angles) * lookat_vector[1],
        np.sin(angles) * lookat_vector[0] + np.cos(angles) * lookat_vector[1]
    ])

    intersections = []
    for ray_direction in ray_directions:
        depth = ray_contour_intersection(camera_pos, ray_direction, contour_x, contour_y)
        if depth != -1:
            intersection_point = camera_pos + depth * ray_direction
            intersections.append(intersection_point)

    return np.array(intersections)

def check_visibility(camera, x_grid, y_grid, fov=np.pi/2, num_rays=128):
    """
    Check if each location in the grid is visible by the camera, and compute the ray index
    that corresponds to the pixel in the camera's image plane.
    
    Parameters:
    - camera: The dictionary containing the camera 'position' and 'lookat' vector.
    - x_grid, y_grid: The x and y coordinates of the grid points.
    - fov: The field of view of the camera in radians.
    - num_rays: The number of rays (pixels) in the camera's image plane.
    
    Returns:
    - visibility_mask: A boolean mask indicating which points are visible.
    - ray_indices: The corresponding ray indices (pixels) in the image plane for each visible point.
    """
    camera_pos = camera['position']
    lookat = camera['lookat']
    
    # Compute vectors from camera to grid points
    vec_to_grid = np.stack([x_grid - camera_pos[0], y_grid - camera_pos[1]], axis=-1)
    
    # Normalize vectors
    vec_norm = np.linalg.norm(vec_to_grid, axis=-1)
    vec_to_grid_normalized = vec_to_grid / vec_norm[..., np.newaxis]
    
    # Compute dot product to check if within FOV
    dot_products = np.sum(vec_to_grid_normalized * lookat, axis=-1)
    visibility_mask = dot_products > np.cos(fov / 2)  # Points within FOV
    
    # Compute angles between the lookat vector and vectors to grid points
    angles_to_grid = np.arctan2(vec_to_grid_normalized[..., 1], vec_to_grid_normalized[..., 0])  # Angles from camera to grid points
    lookat_angle = np.arctan2(lookat[1], lookat[0])  # Angle of the lookat vector

    # Compute relative angles between grid points and the camera's lookat vector
    relative_angles = angles_to_grid - lookat_angle

    # Normalize angles to the range [-pi, pi]
    relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi

    # Points within the FOV are mapped to ray indices (0 to num_rays - 1)
    ray_indices = np.full(relative_angles.shape, -1)  # Initialize with -1 (not visible)
    within_fov_mask = np.abs(relative_angles) <= (fov / 2)  # Check if within FOV
    ray_indices[within_fov_mask] = np.floor(
        ((relative_angles[within_fov_mask] + (fov / 2)) / fov) * num_rays
    ).astype(int)

    return visibility_mask, ray_indices

class Dual2D:
    def __init__(self, real=None, dual=None):
        if real is None:
            real = torch.tensor([1.0, 0.0], dtype=torch.float32, requires_grad=True)
        if dual is None:
            dual = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
            
        self.real = real
        self.dual = dual

    @staticmethod
    def from_transform(angle, translation):
        angle = torch.as_tensor(angle, dtype=torch.float32)
        translation = torch.as_tensor(translation, dtype=torch.float32)
        
        real = torch.tensor([torch.cos(angle), torch.sin(angle)])
        
        dual_x = 0.5 * (translation[..., 0] * real[0] - translation[..., 1] * real[1])
        dual_y = 0.5 * (translation[..., 0] * real[1] + translation[..., 1] * real[0])
        dual = torch.stack([dual_x, dual_y])
        
        return Dual2D(real, dual)

    def get_params(self):
        return [self.real, self.dual]

    def loss_get_norm(self):
        return (1.0 - torch.norm(self.real))**2

    def transform_point(self, point):
        point = torch.as_tensor(point, dtype=torch.float32)
        
        # Normalize real vector before transformation
        real_norm = self.real / torch.norm(self.real)
        
        try:
            # Rotation matrix applied to point
            rotated = torch.stack([
                real_norm[0] * point[..., 0] - real_norm[1] * point[..., 1],
                real_norm[1] * point[..., 0] + real_norm[0] * point[..., 1]
            ], dim=-1)
        except:
            print('eer')
        
        # Translation component using normalized real vector
        translation = 2.0 * torch.stack([
            self.dual[0] * real_norm[0] + self.dual[1] * real_norm[1],
            self.dual[1] * real_norm[0] - self.dual[0] * real_norm[1]
        ], dim=-1)
        
        return rotated + translation

    def get_transform_params_vector(self):
        """Return transform parameters as a single vector"""
        return torch.cat([self.real, self.dual])



class WarpGrid2D:
    def __init__(self, grid_size: Tuple[int, int], bounds: Tuple[float, float, float, float]):
        """
        Initialize a 2D warp grid with Dual quaternions.
        """
        self.grid_size = grid_size
        self.bounds = bounds
        
        x = torch.linspace(bounds[0], bounds[1], grid_size[1])
        y = torch.linspace(bounds[2], bounds[3], grid_size[0])
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.grid_positions = torch.stack([grid_x, grid_y], dim=-1)
        
        self.grid_quaternions = []
        for i in range(grid_size[0]):
            row = []

            for j in range(grid_size[1]):
                row.append(Dual2D())

            self.grid_quaternions.append(row)
            

        self.grid_weight = torch.ones((grid_size[0], grid_size[1]), requires_grad=True)
        
    def get_params(self) -> List[torch.nn.Parameter]:
        """Return all learnable parameters."""
        params = [self.grid_weight]
        for row in self.grid_quaternions:
            for dual in row:
                params.extend(dual.get_params())
        return params
    
    def _compute_weights(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute blending weights for each grid point for given input points.
        """
        points_expanded = points.unsqueeze(-2).unsqueeze(-2)
        grid_positions_expanded = self.grid_positions.unsqueeze(0)
        
        squared_distances = torch.sum(
            (points_expanded - grid_positions_expanded) ** 2, 
            dim=-1
        )
        
        # RBF Kernel
        weights = torch.exp(-squared_distances / (2 * self.grid_weight ** 2))
        
        return weights
    
    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transform input points using the blended grid of Dual quaternions.
        """
        weights = self._compute_weights(points)
        
        sum_real = torch.zeros_like(points)
        sum_dual = torch.zeros_like(points)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                weight = weights[..., i, j].unsqueeze(-1)  # (..., 1)
                dual = self.grid_quaternions[i][j]
                
                sum_real += dual.real * weight
                sum_dual += dual.dual * weight
        
        real_norm = torch.norm(torch.cat([sum_real], 1), dim=1, keepdim=True)
        normalized_real = sum_real / real_norm
        normalized_dual = sum_dual / real_norm
        
        points_transformed = torch.zeros_like(points)
        for i in range(points.shape[0]):
            blended_dual = Dual2D(normalized_real[i], normalized_dual[i])
            print(blended_dual.dual, blended_dual.real)
            points_transformed[i] = blended_dual.transform_point(points[i])

        return points_transformed
    
    def loss_get_norm(self) -> torch.Tensor:
        """Compute norm loss for all grid quaternions."""
        total_loss = torch.tensor(0.0)
        for row in self.grid_quaternions:
            for dual in row:
                total_loss = total_loss + dual.loss_get_norm()
        return total_loss
    
    def arap_loss(self, scale=False):

        total_loss = torch.tensor(0.0)
        

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dual = self.grid_quaternions[i][j]

                neighbors = []
                if i > 0:
                    neighbors.append(self.grid_quaternions[i - 1][j])
                if i < self.grid_size[0] - 1:
                    neighbors.append(self.grid_quaternions[i + 1][j])
                if j > 0:
                    neighbors.append(self.grid_quaternions[i][j - 1])
                if j < self.grid_size[1] - 1:
                    neighbors.append(self.grid_quaternions[i][j + 1])
                

                for neighbor in neighbors:
                    total_loss = total_loss + self._compute_arap_loss(dual, neighbor, scale=scale)

        return total_loss

    def _compute_arap_loss(self, dual1: Dual2D, dual2: Dual2D, scale: bool = False) -> torch.Tensor:

        # Compute rotation difference
        rotation_diff = dual1.real[0] * dual2.real[1] - dual1.real[1] * dual2.real[0]
        rotation_diff = torch.acos(torch.clamp(rotation_diff, -1.0, 1.0))
        
        # Compute translation difference
        translation_diff = 2.0 * torch.stack([
            dual1.dual[0] * dual2.real[0] + dual1.dual[1] * dual2.real[1],
            dual1.dual[1] * dual2.real[0] - dual1.dual[0] * dual2.real[1]
        ], dim=-1)
        
        if(scale):
            return 0.1 * torch.sum(rotation_diff ** 2) + torch.sum(translation_diff ** 2)

        return torch.sum(rotation_diff ** 2) + torch.sum(translation_diff ** 2)


def create_points(num_points=20):
    points_per_side = num_points // 4

    top = torch.stack([
        torch.linspace(-1, 1, points_per_side),
        torch.full((points_per_side,), 0.5)
    ], dim=1)

    bottom = torch.stack([
        torch.linspace(-1, 1, points_per_side),
        torch.full((points_per_side,), -0.5)
    ], dim=1)

    left = torch.stack([
        torch.full((points_per_side,), -1),
        torch.linspace(-0.5, 0.5, points_per_side)
    ], dim=1)

    right = torch.stack([
        torch.full((points_per_side,), 1),
        torch.linspace(-0.5, 0.5, points_per_side)
    ], dim=1)

    return torch.cat([top, bottom, left, right], dim=0)


def create_nonlinearwarp_data(num_points=80):
    points_per_side = num_points // 16 
    rectangles = []
    
    offsets = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)]

    start = []
    end = []
    
    i = 0
    for offset_x, offset_y in offsets:
        top = torch.stack([
            torch.linspace(-0.5, 0.5, points_per_side) + offset_x,
            torch.full((points_per_side,), 0.25) + offset_y
        ], dim=1)

        bottom = torch.stack([
            torch.linspace(-0.5, 0.5, points_per_side) + offset_x,
            torch.full((points_per_side,), -0.25) + offset_y
        ], dim=1)

        left = torch.stack([
            torch.full((points_per_side,), -0.5) + offset_x,
            torch.linspace(-0.25, 0.25, points_per_side) + offset_y
        ], dim=1)

        right = torch.stack([
            torch.full((points_per_side,), 0.5) + offset_x,
            torch.linspace(-0.25, 0.25, points_per_side) + offset_y
        ], dim=1)
        
        if(i == 0 or i==3):
            start.append(torch.cat([top, bottom, left, right], dim=0))
        else:
            end.append(torch.cat([top, bottom, left, right], dim=0))
        i = i + 1
    
    return torch.cat(start, dim=0), torch.cat(end, dim=0)


def matched_mse(transformed_points, target_points):
    return torch.mean(torch.sum((transformed_points - target_points) ** 2, dim=1))

def chamfer_mse(source_points: torch.Tensor, target_points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute Chamfer Mean Squared Error between two point sets."""

    p1 = source_points.unsqueeze(1)  # (N, 1, D)
    p2 = target_points.unsqueeze(0)  # (1, M, D)
    
    distances = torch.sum((p1 - p2) ** 2, dim=2)  # (N, M)
    
    min_dist_1to2 = torch.min(distances, dim=1)[0]  # (N,)
    min_dist_2to1 = torch.min(distances, dim=0)[0]  # (M,)
    
    chamfer_dist = (torch.mean(min_dist_1to2) + torch.mean(min_dist_2to1)) / 2
    
    return chamfer_dist
    


from matplotlib.animation import FFMpegWriter


def warp_grid():

    grid = WarpGrid2D(
        grid_size=(4, 4), 
        bounds=(-7, 7, -7, 7)
    )

    points, target_points = create_nonlinearwarp_data()

    torch.manual_seed(42)
    optimizer = torch.optim.SGD(grid.get_params(), lr=0.02)
    losses = []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    def animate(frame):
        if frame < 1000:
            transformed_points = grid.transform_points(points)
            loss_chamfer = chamfer_mse(transformed_points, target_points)

            loss = loss_chamfer  + 0.002 * grid.arap_loss(scale=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        ax1.clear()
        ax2.clear()
        ax3.clear()
    
        current_points = grid.transform_points(points).detach().numpy()
        target_points_np = target_points.detach().numpy()
    
        ax1.scatter(current_points[:, 0], current_points[:, 1], 
                color='blue', label='Current', s=30, alpha=0.7)
        ax1.scatter(target_points_np[:, 0], target_points_np[:, 1], 
                color='red', label='Target', s=30, alpha=0.3)
    
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.grid(True)
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.set_title(f'Point Transformation\nFrame: {frame}')
    
        if losses:
            ax2.plot(losses)
            ax2.set_yscale('log')
            ax2.grid(True)
            ax2.set_title('Point Matching Loss')
            ax2.set_xlabel('Optimization step')
            ax2.set_ylabel('Loss')

    anim = animation.FuncAnimation(
        fig, animate, frames=120, interval=50, blit=False
    )

    writer = FFMpegWriter(
    fps=20,
    bitrate=1800
    )

    anim.save('animation.mp4', writer=writer)

    plt.show()
    

    visualize_grid_warp(grid)



def visualize_grid_warp(grid: WarpGrid2D, density: int = 50, view_margin: float = 1.0):
    """Visualize how the grid warps 2D space using a normalized vector field plot."""
    # Expand view bounds by margin
    x = np.linspace(grid.bounds[0] - view_margin, grid.bounds[1] + view_margin, density)
    y = np.linspace(grid.bounds[2] - view_margin, grid.bounds[3] + view_margin, density)
    X, Y = np.meshgrid(x, y)
    
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    transformed_points = grid.transform_points(points)
    
    U = transformed_points[:, 0].detach().numpy() - points[:, 0].numpy() / 10
    V = transformed_points[:, 1].detach().numpy() - points[:, 1].numpy() / 10
    
    magnitudes = np.sqrt(U**2 + V**2)
    magnitudes = np.where(magnitudes > 0, magnitudes, 1)
    U = U / magnitudes
    V = V / magnitudes
    
    U = U.reshape(density, density)
    V = V.reshape(density, density)
    
    plt.figure(figsize=(12, 12))
    plt.quiver(X, Y, U, V, scale=30, units='width')
    
    grid_pos = grid.grid_positions.numpy()
    plt.scatter(grid_pos[..., 0], grid_pos[..., 1], c='red', s=100, label='Grid Points')
    
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.grid(True)
    plt.title('Normalized Grid Warp Vector Field')
    plt.legend()
    plt.show()


def run_robot_reconstruction():

    contours_x, contours_y = generate_robot_contour(max_radius=1.2)

    cameras = generate_camera_trajectory(num_points=1, radius_min = 2.5, radius_max=3) # 50

    points = []

    for camera in cameras:

        for i in range(len(contours_x) - 5):
            intersections = calc_rays_intersections(camera, contours_x[i], contours_y[i])
            points.append(intersections)



    # Plot the GT shape
    plt.figure(figsize=(8, 8))
    plt.fill(contours_x[0], contours_y[0], color='red', alpha=0.5, edgecolor='black', linewidth=2, label="GT Shape")

    plot_camera_trajectory_with_FOV(cameras['position'])

    for intersections in points:
        plt.scatter(intersections[:, 0], intersections[:, 1], color='blue', label="Intersection Points")
    plt.title("2D Volume Rendering, Figure 1: Input")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  
    plt.show()





    plt.figure(figsize=(8, 8))
    for intersections in points:

        color = np.random.rand(3,)
        plt.scatter(intersections[:, 0], intersections[:, 1], color=color, label="Intersection Points")
    plt.title("2D Volume Rendering, Figure 1: Input")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()







    def fit_grid(target_points, points, steps=50):


        grid = WarpGrid2D(
            grid_size=(4, 4),
            bounds=(-6, 6, -6, 6) 
        )

        torch.manual_seed(42)

        optimizer = torch.optim.SGD(grid.get_params(), lr=0.1)
        losses = []

        for i in range(150):
            transformed_points = grid.transform_points(points)
            loss_points = chamfer_mse(transformed_points, target_points)

            loss = loss_points + 0.01 * grid.arap_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return grid



    transforms = []
    transformed_sets = [points[0]] # The canonical frame 
    
    for i in range(len(points) - 1):
        target_points = torch.tensor(points[i], dtype=torch.float32)
        source_points = torch.tensor(points[i + 1], dtype=torch.float32)
        w = fit_grid(target_points, source_points, steps=150)
        transforms.append(w)
    
    for i in range(1, len(points)):
        current_points = torch.tensor(points[i], dtype=torch.float32)
        
        for j in range(i):
            current_points = transforms[j].transform_points(current_points)
        
        transformed_sets.append(current_points.detach().numpy())
    
    plt.figure(figsize=(8, 8))
    
    plt.scatter(transformed_sets[0][:, 0], transformed_sets[0][:, 1], 
               color='blue', 
               label="Set 0")
    
    for i, point_set in enumerate(transformed_sets[1:], 1):
        plt.scatter(point_set[:, 0], point_set[:, 1], 
                   color='red', 
                   label=f"Set {i}")
    
    plt.title("2D Volume Rendering, Figure 1: Input")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()



import sys
if __name__ == '__main__':

    if(len(sys.argv) < 2):
        print('Please provide an argument')
        sys.exit(1)
    
    if(sys.argv[1] == '--warp'):
        warp_grid()
    elif(sys.argv[1] == '--robot'):
        run_robot_reconstruction()