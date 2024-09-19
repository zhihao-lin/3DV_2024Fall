import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def viz_demo(R, T, R2, T2, points_3d): 
    print("R", R)
    print("T", T)
    # Define the length of the camera's coordinate axes
    axis_length = 100.0

    # Compute the camera's coordinate axes based on the orientation
    x_axis = R[:, 0] * axis_length
    y_axis = R[:, 1] * axis_length
    z_axis = R[:, 2] * axis_length

    # Define the length of the camera's coordinate axes
    axis_length2 = 50.0

    # Compute the camera's coordinate axes based on the orientation
    x_axis2 = R[:, 0] * axis_length2
    y_axis2 = R[:, 1] * axis_length2
    z_axis2 = R[:, 2] * axis_length2

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2])

    # Plot the camera position
    ax.scatter(T[0], T[1], T[2], color='r', label='Camera Position')

    # Plot the camera's coordinate axes
    ax.quiver(T[0], T[1], T[2], 
            x_axis[0], x_axis[1], x_axis[2], color='b', length=axis_length, label='X Axis')
    ax.quiver(T[0], T[1], T[2], 
            y_axis[0], y_axis[1], y_axis[2], color='g', length=axis_length, label='Y Axis')
    ax.quiver(T[0], T[1], T[2], 
            z_axis[0], z_axis[1], z_axis[2], color='c', length=axis_length, label='Z Axis')

    # Plot the camera position
    ax.scatter(T2[0], T2[1], T2[2], color='r', label='Camera Position')

    # Plot the camera's coordinate axes
    ax.quiver(T2[0], T2[1], T2[2], 
            x_axis2[0], x_axis2[1], x_axis2[2], color='c', length=axis_length, label='X Axis')
    ax.quiver(T2[0], T2[1], T2[2], 
            y_axis2[0], y_axis2[1], y_axis2[2], color='m', length=axis_length, label='Y Axis')
    ax.quiver(T2[0], T2[1], T2[2], 
            z_axis2[0], z_axis2[1], z_axis2[2], color='y', length=axis_length, label='Z Axis')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

    plt.savefig("mycamera1.png")


def depth2vis(depth, maxthresh = 50):
    depthvis = np.clip(depth,0,maxthresh)
    depthvis = depthvis/maxthresh*255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))

    return depthvis

def flow2vis(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = _calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if ( mask is not None ):
        mask = mask > 0
        rgb[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return rgb

def _calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def read_gt_pose(file_path, line_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if line_number <= len(lines):
            return lines[line_number - 1].strip()  # Access the line by index
