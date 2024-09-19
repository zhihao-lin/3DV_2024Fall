import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
from util import viz_demo, depth2vis, flow2vis, read_gt_pose

np.set_printoptions(precision=4, suppress=True)  # Set precision and suppress scientific notation

def backproject_depth_to_3d(depth_map, K):
    """
    Back-project the depth map into 3D points.
    
    Args:
        depth_map: Depth map of the image (H x W).
        K: Camera intrinsic matrix (3 x 3).
        
    Returns:
        points_3d: 3D points (H x W x 3).
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))  # Create a grid of pixel coordinates

    # Convert pixel coordinates to normalized coordinates
    uv_homogeneous = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H x W x 3)
    K_inv = np.linalg.inv(K)

    # Compute 3D points in the camera coordinate system
    points_3d = depth_map[..., None] * (uv_homogeneous @ K_inv.T)  # (H x W x 3)
    
    return points_3d



def warp_pixels_with_optical_flow(u, v, optical_flow):
    """
    Warp pixel coordinates using optical flow.
    
    Args:
        u, v: Pixel coordinates in Image 1.
        optical_flow: Optical flow between Image 1 and Image 2 (H x W x 2).
    
    Returns:
        u2, v2: Warped pixel coordinates in Image 2.
    """
    flow_u = optical_flow[..., 0]
    flow_v = optical_flow[..., 1]

    u2 = u + flow_u
    v2 = v + flow_v

    return u2, v2


def estimate_rigid_transform(points_3d, pixel_2d, K):
    """
    Estimate the rigid transformation (R, T) between 3D points in Image 1 and 2D points in Image 2.
    
    Args:
        points_3d: 3D points from Image 1 (N x 3).
        pixel_2d: Corresponding 2D pixel locations in Image 2 (N x 2).
        K: Camera intrinsic matrix (3 x 3).
    
    Returns:
        rvec: Rotation vector (Rodrigues format).
        tvec: Translation vector.
    """
    # Use solvePnP to estimate the rigid transformation
    success, rvec, tvec = cv2.solvePnP(points_3d, pixel_2d, K, None)
    if not success:
        raise RuntimeError("PnP failed to estimate the rigid transformation.")
    return rvec, tvec


def main(depth_map, optical_flow, K):
    """
    Main function to compute the rigid translation between two images.
    
    Args:
        depth_map: Depth map of Image 1 (H x W).
        optical_flow: Optical flow from Image 1 to Image 2 (H x W x 2).
        K: Camera intrinsic matrix (3 x 3).
    
    Returns:
        rotation: Estimated rotation vector between Image 1 and Image 2.
        translation: Estimated translation vector between Image 1 and Image 2.
    """
    # Step 1: Back-project pixels from Image 1 into 3D space
    points_3d = backproject_depth_to_3d(depth_map, K)

    # Step 2: Warp pixels using optical flow
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    u2, v2 = warp_pixels_with_optical_flow(u, v, optical_flow)
    
    # Filter valid points (ensure flow keeps points in the image)
    valid_idx = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)

    # Select valid points
    points_3d = points_3d[valid_idx].reshape(-1, 3)
    pixel_2d = np.stack([u2[valid_idx], v2[valid_idx]], axis=-1).reshape(-1, 2)

    # Step 3: Estimate the rigid transformation
    rvec, tvec = estimate_rigid_transform(points_3d, pixel_2d, K)

    # Step 4: Output the rotation and translation vector
    return rvec, tvec, points_3d


# Example usage:
if __name__ == "__main__":
    # 1. Loading camera parameters
    fx = 320.0  # focal length x
    fy = 320.0  # focal length y
    cx = 320.0  # optical center x
    cy = 240.0  # optical center y

    width = 640
    height = 480

    K = np.array([[fx, 0, cx],  # Example camera intrinsic matrix
                  [0, fy, cy],
                  [0,  0,  1]])

    # 2. Decide frames to be used, load corresponding data
    idx1 = int(159)
    # idx1 = int(116)
    idx2 = idx1 + 1

    # modify the path
    flow_path = r"D:\abandonedfactory_sample_P001\P001\flow\000{}_000{}_flow.npy".format(idx1, idx2)
    depth_path = r"D:\abandonedfactory_sample_P001\P001\depth_left\000{}_left_depth.npy".format(idx1)
    pose_path = r"D:\abandonedfactory_sample_P001\P001\pose_left.txt"

    pose_1_gt = np.array(read_gt_pose(pose_path, idx1).split(), dtype=np.float64)
    pose_2_gt = np.array(read_gt_pose(pose_path, idx2).split(), dtype=np.float64)

    # Load the .npy file
    flow_data = np.load(flow_path)
    depth_data = np.load(depth_path)
    depth_data = np.clip(depth_data, 0, 11)

    print(flow_data.shape)
    print(depth_data.shape)

    left_depth_vis = depth2vis(depth_data)
    flow_vis = flow2vis(flow_data)
    image_d = Image.fromarray(left_depth_vis)
    image_f = Image.fromarray(flow_vis)
    image_d.save('./output_depth.png')
    image_f.save('./output_flow.png')

    # 3. Use depth of image 1 and optical flow between image 1 and 2, 
    # estimate the rigid transform between them
    rot_12, trans_12, points = main(depth_data, flow_data, K)
    np.save('./point3d_{}.npy'.format(idx1), points)

    # 4. Verify the correctness of estimated RT
    # 4.1 construct the SE(3) matrix of RT
    rotation_matrix_12 = R.from_rotvec(rot_12.flatten()).as_matrix()
    translation_12 = trans_12.flatten()
    RT = np.eye(4)
    RT[:3, :3] = rotation_matrix_12
    RT[:3, 3] = translation_12
    print(RT)


    # 4.2 loading ground truth frame pose
    t1, q1 = pose_1_gt[:3], pose_1_gt[3:]
    t2, q2 = pose_2_gt[:3], pose_2_gt[3:]

    # Construct the SE(3) matrix m_1
    rot_mat1 = R.from_quat(q1).as_matrix()
    m_1 = np.eye(4)
    m_1[:3, :3] = rot_mat1
    m_1[:3, 3] = t1
    print(m_1)

    # Construct the SE(3) matrix m_2
    rot_mat2 = R.from_quat(q1).as_matrix()
    m_2 = np.eye(4)
    m_2[:3, :3] = rot_mat2
    m_2[:3, 3] = t2
    print(m_2)

    # Given ground truth m_1 and estimated RT, calculate est_2
    est_2 = m_1 @ RT
    print(est_2)

    # Extract the translation vector (last column of the first 3 rows)
    translation = est_2[:3, 3]
    
    # Extract the rotation matrix (top-left 3x3 submatrix)
    rotation_matrix = est_2[:3, :3]
    
    # Convert the rotation matrix to a quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()  # returns (x, y, z, w) in that order
    
    pose_2_est = np.concatenate((translation, quaternion))

    print("pose_1_gt", pose_1_gt)
    print("pose_2_gt", pose_2_gt)
    print("pose_2_est", pose_2_est)

    # Save camera poses into a .npy file for visualization
    stacked_array = np.stack((pose_1_gt, pose_2_gt, pose_2_est))
    print("stacked_array", stacked_array, stacked_array.shape)

    np.save('./poses_{}.npy'.format(idx1), stacked_array)



    import os; os._exit(0)



    viz_demo(rotation_matrix1, t1, rotation_matrix2, t2, points)


