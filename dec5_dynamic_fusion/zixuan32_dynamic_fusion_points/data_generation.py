import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Create directories if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('data/frames'):
    os.makedirs('data/frames')

# Argument parser
parser = argparse.ArgumentParser(description="Generate different types of deformations.")
parser.add_argument('--pattern', type=str, choices=['crossing_lines', 'expanding_circle', 'oscillating_ellipse'], required=True,
                    help="Choose the deformation pattern.")
args = parser.parse_args()

# Parameters
num_frames = 25   # Total number of frames
num_points = 150  # Number of points per frame
oversample_factor = 5  # Factor for oversampling points

# Initialize data array
# Shape: [n_frames, n_points, 5] -> [X, Y, R, G, B]
data_array = np.zeros((num_frames, num_points, 5))

# Crossing Lines
def initial_lines(num_points):
    x1 = np.zeros(num_points // 2)
    y1 = np.linspace(-1, 1, num_points // 2)
    x2 = np.linspace(-1, 1, num_points // 2)
    y2 = np.zeros(num_points // 2)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y

def deform_lines(x, y, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    coords = np.stack((x, y), axis=1)
    rotated_coords = coords @ rotation_matrix.T
    return rotated_coords[:, 0], rotated_coords[:, 1]

# Expanding Circle
def initial_circle(num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return x, y

def deform_circle(x, y, scale):
    return scale * x, scale * y

# Oscillating Ellipse
def deform_ellipse(x, y, aspect_ratio):
    return x * aspect_ratio, y / aspect_ratio

# Spiral Motion
def deform_spiral(x, y, t):
    radius = np.sqrt(x**2 + y**2) + 0.1 * t
    angle = np.arctan2(y, x) + 0.1 * t
    x_new = radius * np.cos(angle)
    y_new = radius * np.sin(angle)
    return x_new, y_new

# Sine Wave Transformation
def deform_sine_wave(x, y, amplitude, t):
    y_new = y + amplitude * np.sin(2 * np.pi * x + t)
    return x, y_new

# RGB Mapping Function
def map_rgb(x, y):
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    r = x_norm
    g = y_norm
    b = 0.5 * (np.sin(np.pi * x_norm) + 1)
    rgb = np.stack((r, g, b), axis=1)
    return rgb

# Generate data based on pattern
if args.pattern == 'crossing_lines':
    angle_start = np.pi / 2
    angle_end = 0
    angles = np.linspace(angle_start, angle_end, num_frames)

    for frame_idx, angle in enumerate(angles):
        x0, y0 = initial_lines(num_points * oversample_factor)
        x_def, y_def = deform_lines(x0, y0, angle)
        indices = np.random.choice(len(x_def), num_points, replace=False)
        x_sample = x_def[indices]
        y_sample = y_def[indices]
        rgb = map_rgb(x0[indices], y0[indices])
        data_array[frame_idx, :, 0] = x_sample
        data_array[frame_idx, :, 1] = y_sample
        data_array[frame_idx, :, 2:] = rgb

elif args.pattern == 'expanding_circle':
    scale_start = 0.75
    scale_end = 1.00
    scales = np.linspace(scale_start, scale_end, num_frames)

    for frame_idx, scale in enumerate(scales):
        x0, y0 = initial_circle(num_points * oversample_factor)
        x_def, y_def = deform_circle(x0, y0, scale)
        indices = np.random.choice(len(x_def), num_points, replace=False)
        x_sample = x_def[indices]
        y_sample = y_def[indices]
        rgb = map_rgb(x0[indices], y0[indices])
        data_array[frame_idx, :, 0] = x_sample
        data_array[frame_idx, :, 1] = y_sample
        data_array[frame_idx, :, 2:] = rgb

elif args.pattern == 'oscillating_ellipse':
    aspect_ratios = np.linspace(0.8, 1.2, num_frames)

    for frame_idx, aspect_ratio in enumerate(aspect_ratios):
        x0, y0 = initial_circle(num_points * oversample_factor)
        x_def, y_def = deform_ellipse(x0, y0, aspect_ratio)
        indices = np.random.choice(len(x_def), num_points, replace=False)
        x_sample = x_def[indices]
        y_sample = y_def[indices]
        rgb = map_rgb(x0[indices], y0[indices])
        data_array[frame_idx, :, 0] = x_sample
        data_array[frame_idx, :, 1] = y_sample
        data_array[frame_idx, :, 2:] = rgb

# Save the frames and data
for frame_idx in range(num_frames):
    plt.figure(figsize=(6, 6))
    plt.scatter(data_array[frame_idx, :, 0], data_array[frame_idx, :, 1], c=data_array[frame_idx, :, 2:], s=2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'data/frames/{frame_idx:04d}.png')
    plt.close()

# Save as gif
import imageio
images = []
for i in range(num_frames):
    filename = f'data/frames/{i:04d}.png'
    images.append(imageio.imread(filename))
imageio.mimsave('data/animation.gif', images, duration=0.1, loop=0)

# Save the data array
np.save(f'data/points_{args.pattern}', data_array)
