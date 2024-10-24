import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
import os
import time
import imageio
import cv2 

# Define SDF and Color Network
class SDFColorNetwork(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super(SDFColorNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        # Input layer
        self.input_layer = nn.Linear(2, hidden_dim)
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        # Output layers
        self.sdf_out = nn.Linear(hidden_dim, 1)
        self.color_out = nn.Linear(hidden_dim, 3)  # Output RGB color

    def forward(self, x):
        
        h = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        
        sdf = self.sdf_out(h)  # Allow negative values
        color = torch.sigmoid(self.color_out(h))  # Color values in [0,1]
        
        # x must be in [-1, 1]
        mask = (x[:, 0] >= -1) & (x[:, 0] <= 1) & (x[:, 1] >= -1) & (x[:, 1] <= 1)
        mask = ~mask
        # Avoid in-place operations by using torch.where
        # sdf = torch.where(mask.unsqueeze(-1), torch.tensor(100.0, device=sdf.device), sdf)
        color = torch.where(mask.unsqueeze(-1), torch.tensor(0.0, device=color.device), color)
        
        return sdf, color

# Define volume rendering function
def volume_rendering(rays_o, rays_d, network, num_samples=512, near=0.0, far=2.0, truncation=0.1, rendering_option='baseline'):
    device = rays_o.device
    N_rays = rays_o.shape[0]
    t_vals = torch.linspace(near, far, num_samples, dtype=torch.float32).to(device)  # (N_samples)
    t_vals = t_vals.unsqueeze(0).repeat(N_rays, 1)  # (N_rays, N_samples)
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N_rays, N_samples-1)
    deltas = torch.cat([deltas, torch.full([N_rays, 1], 1e-3, dtype=torch.float32).to(device)], dim=-1)  # (N_rays, N_samples)

    # Sample points
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)  # (N_rays, N_samples, 2)
    pts_flat = pts.reshape(-1, 2)  # (N_rays * N_samples, 2)

    # Predict SDF and color
    sdf_vals, colors = network(pts_flat)  # (N_rays * N_samples, 1), (N_rays * N_samples, 3)
    sdf_vals = sdf_vals.reshape(N_rays, num_samples)  # (N_rays, N_samples)
    colors = colors.reshape(N_rays, num_samples, 3)  # (N_rays, N_samples, 3)

    if rendering_option == 'neuS':
        # NeuS proposed rendering technique
        beta = 0.1  # Hyperparameter, adjust or learn as needed

        # Define phi and cdf functions
        def phi(s, beta):
            return 0.5 * torch.exp(-s.abs() / beta)

        def cdf(s, beta):
            return torch.where(s >= 0, 1 - 0.5 * torch.exp(-s / beta), 0.5 * torch.exp(s / beta))

        # Compute phi and cdf values
        phi_vals = phi(sdf_vals, beta)
        cdf_vals = cdf(sdf_vals, beta)

        # Avoid division by zero
        denom = (1 - cdf_vals).clamp(min=1e-6)

        # Compute densities
        sigma = phi_vals / (beta * denom)

        # Compute alpha values
        alpha = 1 - torch.exp(-sigma * deltas)

        # Compute weights
        T = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=device), 1 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
        weights = alpha * T

    elif rendering_option == 'baseline':
        # Baseline rendering technique using sigmoid mapping
        inv_s = 100.0  # Controls surface sharpness
        sdf_vals_truncated = torch.clamp(sdf_vals, -truncation, truncation)

        # Compute estimated density
        estimated_density = torch.sigmoid(-sdf_vals_truncated * inv_s) * inv_s

        # Compute alpha values
        alpha = 1 - torch.exp(-estimated_density * deltas)

        # Compute weights
        T = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=device), 1 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
        weights = alpha * T

    else:
        raise ValueError("Invalid rendering_option. Choose 'neuS' or 'baseline'.")

    # Aggregate colors
    rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=1)  # (N_rays, 3)

    return rgb_map

def generate_shape_image(shape_type='circle', image_size=256, mnist_data=None, digit=None):
    # Create coordinate grids
    # Generate random noise for each color channel
    xx, yy = np.meshgrid(np.linspace(-np.pi, np.pi, image_size), np.linspace(-np.pi, np.pi, image_size))

    # Generate a randomized color map that is smooth locally but changing globally
    # Random frequencies and phase shifts for smooth variation
    a_r, b_r = np.random.uniform(0.5, 1.5, size=2)
    c_r = np.random.uniform(0, 2*np.pi)
    a_g, b_g = np.random.uniform(0.5, 1.5, size=2)
    c_g = np.random.uniform(0, 2*np.pi)
    a_b, b_b = np.random.uniform(0.5, 1.5, size=2)
    c_b = np.random.uniform(0, 2*np.pi)
    freq = 2
    # Generate color map using sine functions for smooth transitions
    color_map = np.zeros((image_size, image_size, 3))
    color_map[..., 0] = 0.5 + 0.5 * np.sin((a_r * xx + b_r * yy + c_r)*freq)
    color_map[..., 1] = 0.5 + 0.5 * np.sin((a_g * xx + b_g * yy + c_g)*freq)
    color_map[..., 2] = 0.5 + 0.5 * np.sin((a_b * xx + b_b * yy + c_b)*freq)
    color_map = np.clip(color_map, 0, 1)  # Ensure values are within [0, 1]


    if shape_type == 'circle':
        # Create a circle mask
        mask = np.sqrt(xx**2 + yy**2) <= np.pi / 2  # Adjusted for coordinate range
        shape_image = color_map * mask[..., np.newaxis]
    elif shape_type == 'side_circle':
        # Create a circle mask
        mask = np.sqrt(xx**2 + yy**2) <= np.pi / 2
        mask = mask & (xx>0)
        shape_image = color_map * mask[..., np.newaxis]

    elif shape_type == 'square':
        # Create a square mask
        mask = (np.abs(xx) <= np.pi / 2) & (np.abs(yy) <= np.pi / 2)
        shape_image = color_map * mask[..., np.newaxis]
    elif shape_type == 'c_shape':
        # Create a C-like shape mask
        mask = (np.abs(xx) <= np.pi / 2) & (np.abs(yy) <= np.pi / 2)
        inner_mask = (np.abs(xx) <= np.pi / 3) & (np.abs(yy) <= np.pi / 3)
        c_mask = mask & ~inner_mask
        c_mask &= xx < 0.0  # Remove right part to form 'C'
        shape_image = color_map * c_mask[..., np.newaxis]
        mask = c_mask
    elif shape_type == 'multiple_objects':
        # Create multiple object masks (circles)
        mask = np.zeros((image_size, image_size), dtype=bool)
        shape_image = np.zeros((image_size, image_size, 3))
        offsets = [(-np.pi / 2, -np.pi / 2), (np.pi / 2, -np.pi / 2), (0.0, np.pi / 2)]
        for offset in offsets:
            circle = np.sqrt((xx - offset[0])**2 + (yy - offset[1])**2) <= np.pi / 3
            mask |= circle
            shape_image += color_map * circle[..., np.newaxis]
        shape_image = np.clip(shape_image, 0, 1)
    elif shape_type == 'mnist' and mnist_data is not None and digit is not None:
        # Use MNIST digit
        image = mnist_data[digit]
        
        image = cv2.resize(image.numpy(), (image_size, image_size), interpolation=cv2.INTER_AREA)
        # Resize and normalize the image
        image = image / 255.0
        mask = image > 0.1  # Threshold to create mask
        shape_image = color_map * mask[..., np.newaxis]
    else:
        raise ValueError("Invalid shape type or missing MNIST data/digit.")


    return shape_image, mask.astype(np.float32)

# Data generation: Create camera parameters and corresponding 1D views
def generate_data(shape_image, mask, num_cameras=20, num_pixels=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cameras = []
    images = []
    image_size = shape_image.shape[0]
    for i in range(num_cameras):
        angle = i / num_cameras * 2 * np.pi
        # Camera position around the shape
        camera_pos = torch.tensor([np.cos(angle) * 1.5, np.sin(angle) * 1.5], dtype=torch.float32).to(device)
        # Camera looks towards the center
        camera_dir = -camera_pos / torch.norm(camera_pos)
        # Pixel positions on the image plane
        pixel_positions = torch.linspace(-0.5, 0.5, num_pixels, dtype=torch.float32).to(device)
        # Compute ray directions for each pixel
        right = torch.tensor([-camera_dir[1], camera_dir[0]], dtype=torch.float32).to(device)  # Right direction
        rays_o = camera_pos.unsqueeze(0).repeat(num_pixels, 1)  # (N_pixels, 2)
        rays_d = camera_dir.unsqueeze(0).repeat(num_pixels, 1) + (pixel_positions.unsqueeze(1) * right.unsqueeze(0))  # (N_pixels, 2)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize

        # Simulate the 1D view by sampling the shape image along the rays
        t_vals = torch.linspace(0.0, 3.0, 512, dtype=torch.float32).to(device)  # Extended range to ensure intersection
        t_vals = t_vals.unsqueeze(0).repeat(num_pixels, 1)  # (N_pixels, N_samples)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)  # (N_pixels, N_samples, 2)

        # Convert points to image coordinates
        pts_np = pts.cpu().numpy()
        xi = ((pts_np[..., 0] + 1) * (image_size / 2)).astype(np.int32)
        yi = ((1 - (pts_np[..., 1] + 1) / 2) * (image_size - 1)).astype(np.int32)
        # Check for valid indices
        valid = (xi >= 0) & (xi < image_size) & (yi >= 0) & (yi < image_size)
        # Ensure indices are within bounds
        xi = np.clip(xi, 0, image_size - 1)
        yi = np.clip(yi, 0, image_size - 1)
        # Get mask values at sampled points
        mask_samples = mask[yi, xi] * valid
        # Find first intersection with the shape
        hits = mask_samples > 0
        first_hit = hits.argmax(axis=1)  # First intersection along each ray
        hit_mask = hits[np.arange(num_pixels), first_hit]
        # For rays that hit the shape, get the intersection points
        hit_pts = pts[np.arange(num_pixels), first_hit]  # (N_pixels, 2)
        # Initialize colors with background color (black)
        colors = np.zeros((num_pixels, 3), dtype=np.float32)
        # For rays that hit the shape, get the color from the shape image
        valid_hits = hit_mask
        xi_hit = xi[np.arange(num_pixels), first_hit][valid_hits]
        yi_hit = yi[np.arange(num_pixels), first_hit][valid_hits]
        colors[valid_hits] = shape_image[yi_hit, xi_hit]
        colors = torch.from_numpy(colors).to(device)

        # Store the camera parameters and the corresponding colors
        images.append(colors)
        cameras.append({'rays_o': rays_o, 'rays_d': rays_d})

    return cameras, images

# Training the network
def train_network(network, cameras, images, shape_image, mask, num_epochs=2000, lr=5e-4, truncation=0.1):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    device = next(network.parameters()).device
    num_cameras = len(cameras)

    # For video generation
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    frames = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        eikonal_loss = 0.0
        for cam_idx in range(num_cameras):
            rays_o = cameras[cam_idx]['rays_o']  # (N_pixels, 2)
            rays_d = cameras[cam_idx]['rays_d']  # (N_pixels, 2)
            true_colors = images[cam_idx]  # (N_pixels, 3)

            # Sample points along rays
            N_rays = rays_o.shape[0]
            t_vals = torch.linspace(0.0, 2.0, 64, dtype=torch.float32).to(device)  # Adjusted for near and far
            t_vals = t_vals.unsqueeze(0).repeat(N_rays, 1)
            pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)
            pts_flat = pts.reshape(-1, 2).detach().requires_grad_(True)

            # Predict SDF and color
            sdf_vals, colors = network(pts_flat)
            sdf_vals = sdf_vals.reshape(N_rays, -1)
            colors = colors.reshape(N_rays, -1, 3)

            # Truncate SDF values
            sdf_vals_truncated = torch.clamp(sdf_vals, -truncation, truncation)

            # Compute weights
            inv_s = 100.0
            estimated_density = torch.sigmoid(-sdf_vals_truncated * inv_s) * inv_s
            deltas = t_vals[:, 1:] - t_vals[:, :-1]
            deltas = torch.cat([deltas, torch.full([N_rays, 1], 1e-3, dtype=torch.float32).to(device)], dim=-1)
            alpha = 1 - torch.exp(-estimated_density * deltas)
            weights = alpha * torch.cumprod(torch.cat([torch.ones((N_rays, 1), dtype=torch.float32).to(device), 1 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]

            # Aggregate colors
            rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=1)  # (N_rays, 3)

            # Color loss
            color_loss = F.mse_loss(rgb_map, true_colors)
            total_loss += color_loss

            # Eikonal loss
            gradients = torch.autograd.grad(
                outputs=sdf_vals,
                inputs=pts_flat,
                grad_outputs=torch.ones_like(sdf_vals),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradient_norm = gradients.norm(2, dim=1)
            eikonal_loss += ((gradient_norm - 1) ** 2).mean()

        total_loss = total_loss / num_cameras
        eikonal_loss = eikonal_loss / num_cameras
        loss = total_loss + 0.1 * eikonal_loss  # Adjust the weight of the eikonal loss as needed

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Color Loss: {total_loss.item():.6f}, Eikonal Loss: {eikonal_loss.item():.6f}")

            # Generate visualization for video
            sdf_vals_img, pred_mask_img, ground_truth_mask_img = visualize_for_video(network, shape_image, mask)
            sdf_vals_img = np.abs(sdf_vals_img)  # Absolute values for visualization
            # Create a figure for the current frame
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            im1 = axs[0].imshow(sdf_vals_img, cmap='viridis', extent=(-1, 1, -1, 1), origin='lower')
            axs[0].set_title('Predicted SDF Map')
            
            im2 = axs[1].imshow(pred_mask_img, cmap='gray', extent=(-1, 1, -1, 1), origin='lower')
            axs[1].set_title('Predicted Shape')
            im3 = axs[2].imshow(ground_truth_mask_img, cmap='gray', extent=(-1, 1, -1, 1), origin='lower')
            axs[2].set_title('Ground Truth Shape')
            diff_mask_img = np.logical_xor(pred_mask_img, ground_truth_mask_img)
            im4 = axs[3].imshow(diff_mask_img, cmap='gray', extent=(-1, 1, -1, 1), origin='lower')
            # axs[3].imshow(diff_mask_img, cmap='jet', extent=(-1, 1, -1, 1), origin='lower', alpha=0.5)
            axs[3].set_title('Difference ')
            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')
            axs[3].axis('off')
            # Save the current figure to a numpy array
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)


    # Save the video
    
    folder = 'videos'
    name = time.strftime("%Y%m%d-%H%M%S.gif")
    path = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    with imageio.get_writer(path, mode='I', fps=3) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved at: {path}")

# Helper function for visualization during training
def visualize_for_video(network, shape_image, mask):
    device = next(network.parameters()).device
    image_size = shape_image.shape[0]
    x = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y)
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    with torch.no_grad():
        sdf_vals, rgbs = network(pts)
        sdf_vals = sdf_vals.cpu().numpy().reshape(image_size, image_size)
    # transform sdf_vals from (x,y) to (y,x)
    sdf_vals = np.transpose(sdf_vals)
    sdf_vals = sdf_vals[::-1]
    # Predicted shape mask
    pred_mask = sdf_vals <= 0.0
    # Ground truth mask
    ground_truth_mask = mask > 0
    # calculate IOU
    intersection = np.logical_and(pred_mask, ground_truth_mask)
    union = np.logical_or(pred_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)
    print("IOU: ", iou)
    return sdf_vals, pred_mask, ground_truth_mask

# Visualize the reconstructed shape and compare with ground truth
def visualize(network, shape_image, mask):
    device = next(network.parameters()).device
    image_size = shape_image.shape[0]
    # Compute SDF values on a grid
    x = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y)
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    with torch.no_grad():
        sdf_vals, colors = network(pts)
        sdf_vals = sdf_vals.cpu().numpy().reshape(image_size, image_size)
        colors = colors.cpu().numpy().reshape(image_size, image_size, 3)

    # Threshold SDF to get the reconstructed shape mask
    threshold = 0.0  # SDF = 0 corresponds to the surface
    reconstructed_mask = sdf_vals <= threshold

    # Ground truth mask
    ground_truth_mask = mask > 0

    # Compare reconstructed shape with ground truth
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Reconstructed shape
    axs[0].imshow(reconstructed_mask, extent=(-1, 1, -1, 1), origin='lower', cmap='gray')
    axs[0].set_title('Reconstructed Shape Mask')
    # large title
    # large title font size
    
    # no axis
    axs[0].axis('off')

    # Ground truth shape
    axs[1].imshow(ground_truth_mask, extent=(-1, 1, -1, 1), origin='lower', cmap='gray')
    axs[1].set_title('Ground Truth Shape Mask')
    axs[1].axis('off')

    # Overlay comparison
    diff_mask = np.logical_xor(reconstructed_mask, ground_truth_mask)
    axs[2].imshow(ground_truth_mask, extent=(-1, 1, -1, 1), origin='lower', cmap='gray')
    axs[2].imshow(diff_mask, extent=(-1, 1, -1, 1), origin='lower', cmap='jet', alpha=0.5)
    axs[2].set_title('Difference (Red areas indicate mismatch)')
    axs[2].axis('off')

    plt.show()

    # Display reconstructed shape with colors
    reconstructed_image = colors * reconstructed_mask[:, :, np.newaxis]
    plt.figure(figsize=(8, 8))
    plt.imshow(reconstructed_image, extent=(-1, 1, -1, 1), origin='lower')
    plt.title('Reconstructed Shape with Colors')
    
    plt.show()

    # Display ground truth shape with colors
    plt.figure(figsize=(8, 8))
    plt.imshow(shape_image, extent=(-1, 1, -1, 1), origin='lower')
    plt.title('Ground Truth Shape with Colors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True)
    mnist_data = {}
    for i in range(10):
        idx = (mnist_dataset.targets == i).nonzero(as_tuple=True)[0][0]
        mnist_data[i] = mnist_dataset.data[idx]

    # List of shapes to choose from
    # shape_list = ['mnist']
    shape_list = ['side_circle', 'circle', 'square', 'c_shape', 'multiple_objects', 'mnist']
    # shape_list = [ 'multiple_objects', 'mnist']
    for shape_type in shape_list:
        if shape_type == 'mnist':
            for digit in range(10):
                print(f"Processing shape: MNIST digit {digit}")
                shape_image, mask = generate_shape_image(shape_type='mnist', mnist_data=mnist_data, digit=digit)
                run_training(shape_image, mask, shape_name=f"mnist_{digit}")
        else:
            print(f"Processing shape: {shape_type}")
            shape_image, mask = generate_shape_image(shape_type=shape_type)
            run_training(shape_image, mask, shape_name=shape_type)

def run_training(shape_image, mask, shape_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate data
    print("Generating data..., the device is: ", device)
    cameras, images = generate_data(shape_image, mask, num_cameras=50)
    print("Data generation completed!")

    # Initialize network
    network = SDFColorNetwork().to(device)
    # Train network
    print("Starting network training...")
    train_network(network, cameras, images, shape_image, mask, num_epochs=3000)
    print("Training completed!")

    # Visualize reconstructed shape and compare with ground truth
    # visualize(network, shape_image, mask)

if __name__ == "__main__":
    main()
