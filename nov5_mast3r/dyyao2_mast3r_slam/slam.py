from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.geometry import geotrf
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r_visloc.localization import run_pnp
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.post_process import estimate_focal_knowing_depth

import imageio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

def filter_matches(match0, match1, conf0, conf1, thres=1.01):


    matches_conf = (conf0[match0[:,1], match0[:,0]]+conf1[match1[:,1], match1[:,0]]) / 2
    filter = matches_conf >= thres

    return match0[filter], match1[filter]

def slerp(q1, q2, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        t: Interpolation parameter (0 to 1)
    
    Returns:
        Interpolated quaternion
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, negate one of the quaternions
    # This ensures we take the shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate angle between quaternions
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    # Calculate interpolation
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)


def downsample_pointcloud(points, num_samples):
    """
    Downsample a point cloud by randomly selecting a subset of points.

    Parameters:
    points (np.ndarray): Original point cloud, shape (N, 3) where N is the number of points.
    num_samples (int): Number of points to sample.

    Returns:
    np.ndarray: Downsampled point cloud with shape (num_samples, 3).
    """
    # Ensure we don't sample more points than available
    num_samples = min(num_samples, len(points))
    
    # Randomly choose indices without replacement
    indices = np.random.choice(len(points), size=num_samples, replace=False)
    
    # Select the sampled points
    downsampled_points = points[indices]
    
    return downsampled_points

def get_matches(desc1, desc2):
     
     # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    H0, W0 = H0.item(), W0.item()
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    H1, W1 = H1.item(), W1.item()
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # ignore small border around the edge
    valid_matches0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
    valid_matches1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
    valid_matches = valid_matches0 & valid_matches1
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    return matches_im0, matches_im1

def interpolate_poses(start_pose, end_pose, every):
    """
    Interpolate between camera poses using linear interpolation for translation
    and spherical linear interpolation (SLERP) for rotation.
    
    Args:
        poses: List of 4x4 numpy arrays representing camera poses
        every: Number of interpolated poses to generate between each pair of poses
    
    Returns:
        List of interpolated 4x4 pose matrices
    """
    
    interpolated_poses = []
    
    # Extract translations
    start_trans = start_pose[:3, 3]
    end_trans = end_pose[:3, 3]
    
    # Extract rotation matrices and convert to quaternions using scipy
    start_rot = Rotation.from_matrix(start_pose[:3, :3])
    end_rot = Rotation.from_matrix(end_pose[:3, :3])
    
    # Get the quaternions
    start_quat = start_rot.as_quat()  # returns x,y,z,w
    end_quat = end_rot.as_quat()
    
    # Generate interpolation factors
    steps = np.linspace(0, 1, every + 1, endpoint=True)[1:]  # Remove last step to avoid duplicates


    for t in steps:
        # Interpolate translation (linear)
        trans = (1 - t) * start_trans + t * end_trans
        
        # Manual SLERP implementation for quaternions
        dot = np.dot(start_quat, end_quat)
        
        # If the dot product is negative, we need to negate one quaternion
        # This ensures we take the shortest path
        if dot < 0:
            end_quat = -end_quat
            dot = -dot
            
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            interp_quat = start_quat + t * (end_quat - start_quat)
            interp_quat = interp_quat / np.linalg.norm(interp_quat)
        else:
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            interp_quat = (s0 * start_quat) + (s1 * end_quat)
        
        # Convert interpolated quaternion back to rotation matrix
        rot_matrix = Rotation.from_quat(interp_quat).as_matrix()
        
        # Construct interpolated pose matrix
        pose = np.eye(4)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = trans
        
        interpolated_poses.append(pose)
    
    # Add the last pose
    
    return interpolated_poses


if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    scene = "scene0707_00"
    BASE = f"./data/{scene}"
    IMG_DIR = os.path.join(BASE, "color_all")     # IMAGE DIRECTLY 

    every = 30
    all_imgs = sorted(glob(IMG_DIR + "/*.jpg"))

    poses = []
    point_clouds = []

    pcds = []
    imgs = []
    cum = np.eye(4)
    poses = [cum]
    focal = 0

    import time

    start_time = time.time()

    for i in tqdm(range(len(all_imgs))):

        if i + every >= len(all_imgs):
            images, rgbs, new_size, old_size = load_images([all_imgs[i]], size=512)
            imgs.append(rgbs[0])
            if len(imgs) == len(poses):
                break
            continue

        images, rgbs, new_size, old_size = load_images([all_imgs[i], all_imgs[i+every]], size=512)

        imgs.append(rgbs[0])

        if i % every != 0:
            continue

        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
        pp = torch.tensor([new_size[0] // 2, new_size[1] // 2])
        if focal == 0:
            focal = estimate_focal_knowing_depth(output['pred1']['pts3d'], pp, focal_mode='weiszfeld').item()

        K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]])

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        matches_im1, matches_im2 = get_matches(desc1, desc2)

        matches_im1_filtered, matches_im2_filtered = filter_matches(matches_im1, matches_im2, pred1['conf'].squeeze(0).detach(), pred2['conf'].squeeze(0).detach())

        pts3d_1 = pred1['pts3d'][0].numpy()
        pts3d_2 = pred2['pts3d_in_other_view'][0].numpy()

        # run PnP
        pts_3d = pts3d_1[matches_im1_filtered[:, 1], matches_im1_filtered[:, 0]]
        pts_2d = matches_im2_filtered.astype(np.float32)

        success, RT = run_pnp(pts_2d, pts_3d, K, mode='cv2')   #cam2world

        if not success:
            breakpoint()

        if len(pcds) == 0:
            pcds.append(geotrf(cum, pts3d_1))

        cums = interpolate_poses(cum, cum@RT, every)

        poses.extend(cums)

        pcds.append(geotrf(cum, pts3d_2))

        cum = cum @ RT

    end_time = time.time()

    print((end_time - start_time) / len(all_imgs))

    poses = np.array(poses)
    imgs = np.array(imgs)
    pcds = np.array(pcds)

    output_dict = {  # DISP is saved in opt_shape
        'pcds': pcds,
        'poses': poses,
        'imgs': imgs,
        'K': K
    }

    np.savez(f"output_{scene}_{every}_interpolated_test.npz", **output_dict)