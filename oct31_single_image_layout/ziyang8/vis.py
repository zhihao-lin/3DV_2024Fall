import os
import numpy as np
import torch
import imageio
import math
import matplotlib.pyplot as plt

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    AmbientLights,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle


def vis_depth(depth, cmap='magma_r'):
    cmap = plt.get_cmap(cmap)
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cmap(depth)
    depth = (depth[:,:,:3]*255).astype(np.uint8)
    return depth

def vis_normal(normal):
    normal = normal.permute(1, 2, 0).cpu().numpy()
    normal = (normal + 1) / 2.0
    normal = (normal * 255).astype(np.uint8)
    return normal

def vis_gradient(grad, cmap='magma_r'):
    grad = grad.cpu().numpy()
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    grad = plt.get_cmap(cmap)(grad)
    grad = (grad[:,:,:3]*255).astype(np.uint8)
    return grad


def compute_fov(focal_length_px, image_size_px):
    return 2 * np.arctan(0.5 * image_size_px / focal_length_px)

def render_spiral(mesh, output_path, num_frames=300, start_distance=0, forward_distance=3, spiral_radius=10, focal_length_px=500, W=1024, H=1024):
    os.makedirs(output_path, exist_ok=True)
    device = mesh.device

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = AmbientLights(device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=None,
            lights=lights
        )
    )

    frames = []
    for i in range(num_frames):
        # Compute interpolation factor for forward movement
        t = i / num_frames

        current_distance = start_distance + t * forward_distance
        spiral_angle = t * 5 * math.pi  
        radius = min(1, 10*t) * spiral_radius 

        eye_position = torch.tensor([
            [radius/100 * math.cos(spiral_angle), 
            radius/100 * math.sin(spiral_angle), 
            current_distance]  
        ])

        look_at_position = torch.tensor([
            [radius * math.cos(spiral_angle),  
            radius * math.sin(spiral_angle),  
            100.0]  
        ])
        
        R, T = look_at_view_transform(
            eye=eye_position, 
            at=look_at_position, 
            up=torch.tensor([[0.0, -1.0, 0.0]])
        )

        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
        )

        # Update renderer's cameras
        renderer.rasterizer.cameras = cameras
        renderer.shader.cameras = cameras

        # Render the image
        images = renderer(mesh)
        image = images[0, ..., :3].cpu().numpy()
        image = (image * 255).astype(np.uint8)
        frames.append(image)

        # Save individual frames (optional)
        # imageio.imsave(os.path.join(output_path, f"frame_{i:03d}.png"), image)
        if i % 10 == 0 or i == num_frames - 1:
            print(f"Rendered frame {i + 1}/{num_frames}")

    # Save as a GIF
    gif_path = os.path.join(output_path, "forward_spiral_render.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Forward spiral rendering saved as GIF at {gif_path}")

    # Optionally, save as a video
    video_path = os.path.join(output_path, "forward_spiral_render.mp4")
    writer = imageio.get_writer(video_path, fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Forward spiral rendering saved as video at {video_path}")

def render_orbit(mesh, output_path, num_frames=300, start_distance=0, forward_distance=3, spiral_radius=10, focal_length_px=500, W=1024, H=1024):
    os.makedirs(output_path, exist_ok=True)
    device = mesh.device

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = AmbientLights(device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=None,
            lights=lights
        )
    )
    
    half_frames = num_frames // 2
    frames = []
    for i in range(num_frames):
        if i < half_frames:
            t = (i / half_frames) * 2 * math.pi 
            elev = 15 * math.sin(t)  
            azim = 0  
        else:
            t = ((i - half_frames) / half_frames) * 2 * math.pi  
            elev = 0  # Reset elev to 0
            azim = 15 * math.sin(t)  
            
        up_vector = torch.tensor([[0.0, -1.0, 0.0]], device=device)  # Standard up direction
        at_vector = torch.tensor([[0.0, 0.0, 1]], device=device)  # Look at the origin
        dist = 1.5  # Adjust as needed
        R, T = look_at_view_transform(
            dist=dist,
            elev=elev,
            azim=azim+180,
            at=at_vector,
            up=up_vector
        )

        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
        )

        # Update renderer's cameras
        renderer.rasterizer.cameras = cameras
        renderer.shader.cameras = cameras

        # Render the image
        images = renderer(mesh)
        image = images[0, ..., :3].cpu().numpy()
        image = (image * 255).astype(np.uint8)
        frames.append(image)

        # Save individual frames (optional)
        # imageio.imsave(os.path.join(output_path, f"frame_{i:03d}.png"), image)
        if i % 10 == 0 or i == num_frames - 1:
            print(f"Rendered frame {i + 1}/{num_frames}")

    # Save as a GIF
    gif_path = os.path.join(output_path, "orbit_render.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Orbit rendering saved as GIF at {gif_path}")

    # Optionally, save as a video
    video_path = os.path.join(output_path, "orbit_render.mp4")
    writer = imageio.get_writer(video_path, fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Orbit rendering saved as video at {video_path}")