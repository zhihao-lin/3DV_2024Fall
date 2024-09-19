# Instructions
## Hacker (ozgurk2@illinois.edu)
First download the dataset using the script co3d/co3d/download_dataset.py from CO3Dv2 Dataset. 

Then, preprocess the dataset to make it compatible with the training script. preprocess_co3d.py

After, change the paths in the run_training.py and run that training script, it will generate images of the scene and plucker representations for each epoch.

To convert the output images into a gif, use pil_images_to_gif.py.