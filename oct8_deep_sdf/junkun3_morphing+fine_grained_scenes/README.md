# DeepSDF Hacker 1: Shape Morphing + Fine-Grained Scenes

### Hacker: Junkun Chen (junkun3@illinois.edu)

## Task

This hacker task contains two demos:

- Demo 1: Self-adaptive shape morphing via latent interpolation
- Demo 2: Extension of DeepSDF on more fine-grained scenes
	- Large-scale indoor scenes
	- Fine-grained object-centric scenes

## Usage

Both demos are implemented based on this codebase: https://github.com/maurock/DeepSDF/. Please refer to the `README.md` in each demo's folder for installation and some basic operations. 

The code under each demo has some slight different, while the demo 2 was done after demo 1 and therefore is more complete. Please preferably use demo 2's code if you want to do something new with the code.

### How to add a custom shape?

- Please work on the code of demo 2.
- The shape should be `.obj`. If you have other formats, you can convert using Blender.
- Put it as `data/{DATASET}/{Type}/{Object}/models/model_normalized.obj`
  - E.g., `data/NeRFSyn/chair/chair/models/model_normalized.obj`
- Run `python data/normalize_obj.py data/{DATASET}/{Type}/{Object}/models/model_normalized.obj` to normalize it to [-1, 1].
- Run `python data/extract_sdf.py` to construct training samples. You may need to modify the name of the dataset in the code.
- Use it for training.

### How to create a morphing video?
- Train a DeepSDF instance on a dataset containing all the scenes you want to use for morphing. You can construct your own dataset with the above instructions.
  - The checkpoint will be stored at `results/runs_sdf/{CKPT_NAME}`.
- Configure `config_files/reconstruct_from_latent.yaml`
  - Indicate the `{CKPT_NAME}` at `folder_sdf`
  - List the morphing sequence in `obj_ids`, in the format of `{Type}/{Object}` for each, e.g., `chair/chair`.
- Run `scripts/morph_latents.py` to generate the morphing objects. They are saved under `results/runs_sdf/{CKPT_NAME}/meshes_training`.
- Run `scripts/render_morphing_video.py` to render the morphing video and save it under `results/`. You may want to modify the mesh folder and output file name in the code.

### How to export a NeRF dataset to .obj files (if without ground truth)?

- Install NeRFStudio: https://docs.nerf.studio/
- Train a NeRFStudio's NeRF model on the NeRF dataset.
- Use either TSDF or Poisson reconstruction according to the instructions on https://docs.nerf.studio/quickstart/export_geometry.html.