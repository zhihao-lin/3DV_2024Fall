import os
import numpy as np
import rerun as rr
import imageio
import cv2
from util import umeyama_alignment
import torch
# 

from scipy.spatial.transform import Rotation as scipy_R

rr.init("vis")

rr.spawn()

rr.log("/", timeless=True)

def get_scale(traj1, traj2):

    """
    Turn traj1 to traj2
    """

    differences1 = np.diff(traj1, axis=0)
    differences2 = np.diff(traj2, axis=0)

    # Calculate the Euclidean distance for each consecutive point pair
    distances1 = np.linalg.norm(differences1, axis=1)
    distances2 = np.linalg.norm(differences2, axis=1)

    # Sum all distances to get the total length of the path
    total_length1 = np.sum(distances1)
    total_length2 = np.sum(distances2)

    return total_length2 / total_length1

def display_arrows(dir, Rs, ts, color):

    rr.set_time_seconds("stable_time", 0)

    origins = np.repeat(ts, 3, axis=0)

    Rs = Rs.transpose(0, 2, 1).reshape(-1, 3)

    colors = np.array([[[255,0,0], [0,255,0,], [0,0,255]] for _ in range(len(Rs)//3)]).reshape(-1, 3)

    rr.log(f"{dir}/pose", rr.Arrows3D(origins=origins, vectors=Rs*0.001, colors=colors))

    rr.log(f"{dir}", rr.LineStrips3D([ts], colors = color))

def log_info(dir, R, t, K, pcd, rgbs, rgb,vis_image=True):

    

    rr.log(f"{dir}/camera", rr.Transform3D(translation=t, mat3x3=R))

    rr.log(
        f"{dir}/camera",
        rr.Pinhole(
            resolution=[rgbs.shape[2], rgbs.shape[1]],
            image_from_camera=K,
        ),
    )

    rr.log(f"{dir}/points3d", rr.Points3D(pcd.reshape(-1, 3), colors=rgbs.reshape(-1, 3), radii=0.01))
    if vis_image:
        rr.log(f"{dir}/image/rgb", rr.Image(rgb))

def log_cameras(dir, Rs, ts):
    for i, (R, t) in enumerate(zip(Rs, ts)):
        rr.set_time_seconds("stable_time", i)
        rr.log(f"{dir}/camera_{i}", rr.Transform3D(translation=t, mat3x3=R))

def log_control_points(control_points, vis, title, norms=None, colors=None, inlier_filter=None):

    for t in range(len(vis)):

        colors = np.repeat([[255, 0, 0]], len(control_points), axis=0)
        colors[vis[t]] = np.repeat([[0, 255, 0]], np.sum(vis[t]), axis=0)

        rr.set_time_seconds("stable_time", t)
        rr.log(f"control_points_{title}", rr.Points3D(control_points, colors=colors, radii=0.01))

def load_mesh(path, name, rot=None, scale=None, trans=None):

    import open3d as o3d

    print("loading mesh")
    mesh = o3d.io.read_triangle_mesh(path)
    print("finish loading mesh")
    mesh_vertices = np.array(mesh.vertices)

    if rot is not None:
        # mesh_vertices = (blender_2_rdf @ mesh_vertices.T).T

        mesh_vertices = (scale * (rot @ mesh_vertices.T + trans)).T

    rr.log(
        f"world/mesh_{name}",
        rr.Mesh3D(
            vertex_positions=mesh_vertices,
            vertex_colors=mesh.vertex_colors,
            triangle_indices=mesh.triangles,
        ),
        static=True,
    )


scene = "scene0707_00"
gt_poses = np.loadtxt(f"pose.txt")   # GT PATH
every = 30

path = f"./output_{scene}_{every}_interpolated_test.npz"   # RESULTS PATH

results = np.load(path)
pcds = results["pcds"]
poses = results["poses"]
rgbs = results["imgs"]
K = results["K"]
n = poses.shape[0]

gt_poses = gt_poses.reshape(-1, 4, 4)[:n]

h,w = pcds.shape[1:3]

Rs_gt = []
ts_gt = []

Rs = []
ts = []

for i in range(n):

    print(f"Processing {i}/{len(pcds)}", end="\r", flush=True)

    gt_pose = gt_poses[i]

    r = gt_pose[:3, :3]
    t = gt_pose[:3, 3]

    Rs_gt.append(r)
    ts_gt.append(t)

    Rs.append(poses[i][:3,:3])
    ts.append(poses[i][:3, 3])

ts=np.array(ts).T
ts_gt=np.array(ts_gt).T

Rs=np.array(Rs)
Rs_gt=np.array(Rs_gt)

trans = ts[:, 0] - ts_gt[:, 0]
ts_gt = ts_gt + trans[:, None]

rot = Rs[0] @ Rs_gt[0].T
Rs_gt = np.array([rot @ R for R in Rs_gt])
ts_gt = rot @ ts_gt

for i in range(n):

    print(f"Processing {i}/{len(pcds)}", end="\r", flush=True)

    pcd = pcds[i//every]
    rgb = rgbs[i//every]

    R = poses[i][:3, :3]
    t = poses[i][:3, 3]

    rr.set_time_seconds("stable_time", i)
    log_info("pred", Rs[i], ts[:,i], K, pcds[:i//every+1], rgbs[:i+1][::every], rgbs[i])

rr.set_time_seconds("stable_time", 0)
log_info("full", Rs[0], ts[:,0], K, pcds[::30//every], rgbs[::every][::30//every], rgb=None, vis_image=False)

display_arrows("pred_cam", Rs, ts.T, (255,0,0))
display_arrows("gt_cam", Rs_gt, ts_gt.T, (0,0,255))