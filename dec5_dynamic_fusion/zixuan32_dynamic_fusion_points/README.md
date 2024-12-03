## Point Cloud Warp Field Estimation

This project demonstrates the estimation of warp fields in 2D point clouds using dynamic deformation models. It supports various deformation patterns and provides cumulative and forward warp visualizations.

### Data Generation

Run the data generator with your chosen pattern:
```sh
python data_generation.py --pattern [crossing_lines|expanding_circle|oscillating_ellipse]
```

### Warp Field Estimation

Run the estimator on the corresponding data that you generated:
```sh
python warp_field_estimation.py --pattern [crossing_lines|expanding_circle|oscillating_ellipse] --regularization 1.0
```
The code is written with pure python and numpy without GPU acceleration. It will take a while to estimate the wrap fields for all frames.