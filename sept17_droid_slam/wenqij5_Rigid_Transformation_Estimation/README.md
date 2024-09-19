<!-- 
### Set up
- Conda is recommended to manage Python packages
- Install `numpy`, `opencv`, `matplotlib`

### How to run?
- Run `Volume Fusion 2D.ipynb`

Here's a basic structure for your README file. You can adapt this to your project depending on the level of detail and additional instructions you'd like to provide.

--- -->

# Rigid Transformation Estimation via Optical Flow and Depth

### Hacker: Wenqi Jia (wenqij5@illinois.edu)

## Task

This project implements a method to estimate the rigid transformation (rotation and translation) between two images using optical flow and depth information. The method leverages the 3D point correspondences from the depth map of the first image and optical flow to compute the camera's pose change between two frames.


## Set up the environment:

```bash
conda env create -f environment.yml
conda activate open3denv
```

## Usage

1. Download and unzip [Tartanair](https://cmu.box.com/s/5ycmyx1q3vumesl0bozfze1a54ejwgmq) sample data

2. Update loading path in `estimate_rigid_transform.py`

3. Run the transformation estimation code:

```bash
visualization.ipynb
```

## Visualization

To visualize the estimated pose in 3D, run the following:

```bash
python visualize_pose.py
```

This will generate a 3D plot displaying the transformation between the frames.
