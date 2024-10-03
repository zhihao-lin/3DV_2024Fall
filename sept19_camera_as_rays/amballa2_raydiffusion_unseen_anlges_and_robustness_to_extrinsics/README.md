# Hacker - Chaitanya - amballa2@illinois.edu


First download the dataset using the script co3d/co3d/download_dataset.py from CO3Dv2 Dataset.

Then, preprocess the dataset to make it compatible with the training script. preprocess_co3d.py

Again, make sure, paths are correct. 


You can do the following two experiments.

## Exp 1:

Can the paper generalize well for unseen angles? 

To do this, we use 77 images are used for training, and 25 unseen angles for testing

Then train the Diffusion model trained from 8 random images every batch with ```python train.py```. This will take the corect config file. You can change the config if need as you wish.

Make sure to check the unseen test angles are not present in your train_dataloader. Printing the images in your batch would do this. You can change the test angles for a different dataset or even for the chair dataset at co3d_v2.py in Co3dDataset class at line 241 as you need.

To check the predicted cameras, run ```python demo.py``` with the changed paths and correct ckpts in main().

To get the metrics, run ```python eval_jobs.py``` with the correct paths.

## Exp 2:

Is it robust for corruptions in camera extrinsics?	

To do this experiment, select a subset of your data samples, say 8 (you can find them in ```data/8chairs_equal_seperated``` if you directly wish to use this)

Add your choie of noise variance to the ground truth extrinsics at line 247 in ```co3d_v2.py```. 

To check the predicted cameras, run ```python demo.py``` with the changed paths and correct ckpts in main().

To get the metrics, run ```python eval_jobs.py``` with the correct paths.
