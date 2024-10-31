import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import os
import imageio
from depth2normal import Depth2normal
from vis import vis_depth, vis_normal, vis_gradient

# depth to normal through plane svd
def depth2normal(depth, focal_length, window_size=3, device='cuda'):
    if not torch.is_tensor(depth):
        depth = torch.tensor(depth, dtype=torch.float32, device=device)
    else:
        depth = depth.to(device=device, dtype=torch.float32)
    b, h, w = 1, depth.shape[0], depth.shape[1]
    cx, cy = w / 2.0, h / 2.0  

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'  
    )

    x = (x - cx) / focal_length
    y = (y - cy) / focal_length
    z = depth

    points_x = x * z
    points_y = y * z
    points_z = z
    points = torch.stack([points_x, points_y, points_z], dim=0)  # (3, H, W)
    points = points.unsqueeze(0)  # (1, 3, H, W)

    depth_to_normal = Depth2normal(
        d_min=0.0,
        d_max=100.0,
        k=window_size,
        d=1,
        min_nghbr=4,
        gamma=0.05,
        gamma_exception=True
    ).to(device)

    normals, valid_mask = depth_to_normal(points)

    return normals.squeeze()

def get_sobel_kernels(device, dtype):
    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=dtype, device=device)
    
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], dtype=dtype, device=device)
    return sobel_x, sobel_y

def compute_gradient(input_tensor):
    if input_tensor.dim() == 2:
        # (H, W) -> (1, H, W)
        input_tensor = input_tensor.unsqueeze(0)
    elif input_tensor.dim() == 3:
        if input_tensor.size(0) not in [1, 3]:
            raise ValueError("Input tensor must have 1 or 3 channels in shape (C, H, W)")
    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    C, H, W = input_tensor.shape

    device = input_tensor.device
    dtype = input_tensor.dtype

    sobel_x, sobel_y = get_sobel_kernels(device, dtype)
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    input_tensor = input_tensor.unsqueeze(0)
    grad_x = F.conv2d(input_tensor, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(input_tensor, sobel_y, padding=1, groups=C)

    grad_x = grad_x.squeeze(0)  # Shape: (C, H, W)
    grad_y = grad_y.squeeze(0)  # Shape: (C, H, W)

    if input_tensor.shape[1] == H and input_tensor.shape[2] == W and input_tensor.shape[0] == 1:
        grad_x = grad_x.squeeze(0)
        grad_y = grad_y.squeeze(0)

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8) 
    gradient_map = grad_magnitude.sum(0)

    edge_width = 1 # kernel size // 2
    mask = torch.ones_like(gradient_map, device=device, dtype=dtype)
    mask[:edge_width, :] = 0  # Top edges
    mask[-edge_width:, :] = 0  # Bottom edges
    mask[:, :edge_width] = 0  # Left edges
    mask[:, -edge_width:] = 0  # Right edges

    gradient_map = gradient_map * mask
    return gradient_map

    

def get_edge_mask(depth_grad, threshold=0.1):
    depth_grad = (depth_grad - depth_grad.min()) / (depth_grad.max() - depth_grad.min())
    edge_mask = depth_grad > threshold
    return edge_mask

def identify_quads_to_keep(H, W, edge_mask):
    num_quads = (H - 1) * (W - 1)
    device = edge_mask.device
    
    # Generate indices for quads
    row = torch.arange(0, H - 1, device=device).repeat(W - 1)
    col = torch.arange(0, W - 1, device=device).repeat_interleave(H - 1)
    top_left = row * W + col  # Shape: (num_quads,)
    
    # Define corner indices for each quad
    tl = top_left
    tr = top_left + 1
    bl = top_left + W
    br = top_left + W + 1
    
    # Check edges: top (tl-tr), right (tr-br), bottom (br-bl), left (bl-tl)
    edge_top = edge_mask[row, col + 1]      # Between tl and tr
    edge_right = edge_mask[row + 1, col + 1]  # Between tr and br
    edge_bottom = edge_mask[row + 1, col]    # Between br and bl
    edge_left = edge_mask[row, col]          # Between bl and tl
    
    # Determine which quads to keep (no high gradients on any edge)
    quad_keep_mask = ~(edge_top | edge_right | edge_bottom | edge_left)
    
    return quad_keep_mask, tl, tr, bl, br

def export_mesh(mesh, path):
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    vertex_colors = mesh.textures.verts_features_list()[0]

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu().numpy())
    o3d.io.write_triangle_mesh(path, mesh_o3d)

def save_results(image, depth, normal, mesh, output_path):
    depth_vis = vis_depth(depth)
    normal_vis = vis_normal(normal)
    imageio.imsave(os.path.join(output_path, "input.png"), image.permute(1, 2, 0).cpu().numpy())
    imageio.imsave(os.path.join(output_path, "depth.png"), depth_vis)
    imageio.imsave(os.path.join(output_path, "normal.png"), normal_vis)
    export_mesh(mesh, os.path.join(output_path, "mesh.ply"))