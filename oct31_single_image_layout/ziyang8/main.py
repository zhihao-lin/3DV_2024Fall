import os
import torch 
import numpy as np
import imageio
import argparse
import open3d as o3d
import huggingface_hub
from PIL import Image
from utils import depth2normal, export_mesh, save_results, compute_gradient, identify_quads_to_keep, get_edge_mask
from vis import vis_depth, vis_normal, vis_gradient, render_spiral, render_orbit
from depth_model import MODEL_BUILD
from skimage.morphology import skeletonize

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def predict_depth(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_BUILD[args.model_name](device)

    image = Image.open(args.img_path)
    W, H = image.size
    if W > H:
        image = image.resize((1024, int(1024 * H / W)))
    else:
        image = image.resize((int(1024 * W / H), 1024))

    image = np.array(image)[:,:,:3]
    image = torch.from_numpy(image).permute(2, 0, 1).to(device)
    if args.model_name == "depth_pro":
        rgb = image / 255.0
        rgb = rgb * 2 - 1
    predictions = model.infer(rgb)
    
    focal = predictions["focallength_px"]
    depth = predictions["depth"]
    depth = torch.clamp(depth, 0.1, 100)
    focal = focal.item()
    return image, depth, focal

def create_mesh_sheet(image, depth, focal, edge_mask=None, device='cuda'):
    H, W = depth.shape
    rgb = image.permute(1, 2, 0) / 255.0

    # Generate grid offsets
    grid_x = torch.linspace(-1, 1, W, device=device)
    grid_y = torch.linspace(-1, 1, H, device=device)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy') 
    grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  
    grid = grid.reshape(1, H, W, 2)

    z = depth.flatten()  
    x = z * grid_x.flatten() * (W / (2 * focal))
    y = z * grid_y.flatten() * (H / (2 * focal))
    vertices = torch.stack([x, y, z], dim=-1)  
    vertex_colors = rgb.cpu()
    vertex_colors = vertex_colors.reshape(-1, 3)

    if edge_mask is None:
        num_quads = (H - 1) * (W - 1)
        faces = torch.zeros((num_quads * 2, 3), dtype=torch.long, device=device)
        row = torch.arange(0, H - 1, device=device).repeat(W - 1)
        col = torch.arange(0, W - 1, device=device).repeat_interleave(H - 1)
        tl = row * W + col  # Shape: (num_quads,)
        tr = tl + 1
        bl = tl + W
        br = tl + W + 1
    else: 
        quad_keep_mask, tl, tr, bl, br = identify_quads_to_keep(H, W, edge_mask)
        keep_indices = quad_keep_mask.nonzero(as_tuple=False).squeeze()
        tl = tl[keep_indices]
        tr = tr[keep_indices]
        bl = bl[keep_indices]
        br = br[keep_indices]


    tri1 = torch.stack([tl, tr, bl], dim=1)  # Shape: (num_quads, 3)
    tri2 = torch.stack([tr, br, bl], dim=1)  # Shape: (num_quads, 3)
    faces = torch.cat([tri1, tri2], dim=0)

    textures = TexturesVertex(verts_features=[vertex_colors.to(device)])
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    return mesh

# For comparison
def poisson_mesh_recon(image, depth, focal, device='cuda'):
    H, W = depth.shape
    rgb = image.permute(1, 2, 0) / 255.0

    # Generate grid offsets
    grid_x = torch.linspace(-1, 1, W, device=device)
    grid_y = torch.linspace(-1, 1, H, device=device)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy') 
    grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  
    grid = grid.reshape(1, H, W, 2)

    z = depth.flatten()  
    x = z * grid_x.flatten() * (W / (2 * focal))
    y = z * grid_y.flatten() * (H / (2 * focal))
    points = torch.stack([x, y, z], dim=-1)  
    colors = rgb.reshape(-1, 3)

    # Create a mesh with poisson reconstruction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
    pcd.normalize_normals()

    # Perform Poisson reconstruction
    print("Performing Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    
    # Optionally, remove low-density vertices to clean up the mesh
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)  # Remove the lowest 1% density vertices
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def run(args):
    image, depth, focal = predict_depth(args)
    depth = torch.clamp(depth, 0.1, 80)
    normal = depth2normal(depth, focal)
    img_name = args.img_path.split("/")[-1].split(".")[0]
    output_path = os.path.join(args.output_path, img_name)
    os.makedirs(output_path, exist_ok=True)

    # mesh = poisson_mesh_recon(image, depth, focal)
    # o3d.io.write_triangle_mesh(os.path.join(output_path, "poisson_mesh.ply"), mesh)

    inv_depth_grad = compute_gradient(1/(depth+1e-6))
    edge_mask = get_edge_mask(inv_depth_grad, 0.2)
    edge_mask_skel = skeletonize(edge_mask.cpu().numpy())
    edge_mask_skel = torch.from_numpy(edge_mask_skel).to(inv_depth_grad.device)

    inv_depth_grad_vis = vis_gradient(inv_depth_grad)
    imageio.imsave(os.path.join(output_path, "depth_grad.png"), inv_depth_grad_vis)
    imageio.imsave(os.path.join(output_path, "edge_mask.png"), (edge_mask_skel.cpu().numpy() * 255).astype(np.uint8))

    mesh = create_mesh_sheet(image, depth, focal, edge_mask_skel)

    H, W = depth.shape
    save_results(image, depth, normal, mesh, output_path)
    # render_spiral(mesh, output_path, focal_length_px=focal, W=W, H=H)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("--model_name", type=str, default="depth_pro")
    parser.add_argument("--output_path", type=str, default="./output")
    args = parser.parse_args()

    run(args)