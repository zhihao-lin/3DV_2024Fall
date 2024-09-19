# Rigid Transformation Estimation via Optical Flow and Depth

### Hacker: Wenqi Jia (wenqij5@illinois.edu)

## Task

This project implements a method to estimate the rigid transformation (rotation and translation) between two images using optical flow and depth information. 


## Set up the environment:

```bash
conda env create -f environment.yml
conda activate open3denv
```

## Usage

1. Download and unzip [Tartanair]([URL](https://cmu.box.com/s/5ycmyx1q3vumesl0bozfze1a54ejwgmq)) sample data

2. Update loading path in `estimate_RT.py`

3. Run the transformation estimation code:

```bash
python estimate_RT.py
```

## Visualization

To visualize the estimated pose in 3D, run the following:

```bash
python visualize_pose.py
```

This will generate a 3D plot displaying the poses of two views, together with the point clouds.
