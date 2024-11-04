## Quick Start:

1. Clone the [InstantSplat](https://github.com/NVlabs/InstantSplat) git repo (Dust3R+3dgs) and follow its set-up instructions
```git clone --recursive https://github.com/NVlabs/InstantSplat.git```
2. Clone the [Mast3R](https://github.com/naver/mast3r) git repo and place it under ```./submodule```
```
cd InstantSplat/submodule
git clone --recursive https://github.com/naver/mast3r
cd ..
```
3. Copy the ```run_train_infer_mast3r.sh``` to ```./script```
```
cp path/to/run_train_infer_mast3r.sh ./script
```
4. Copy the ```coarse_init_infer_mast3r.py``` to InstantSplat base directory
```
cp path/to/coarse_init_infer_mast3r.py .
```
5. Modify the inputs in ```run_train_infer_mast3r.sh```
6. Run Mast3R + 3dgs
```
bash ./script/run_train_infer_mast3r.sh
```

## Reference
```
@article{leroy2024grounding,
  title={Grounding Image Matching in 3D with MASt3R},
  author={Leroy, Vincent and Cabon, Yohann and Revaud, J{\'e}r{\^o}me},
  journal={arXiv preprint arXiv:2406.09756},
  year={2024}
}

@article{fan2024instantsplat,
  title={Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds},
  author={Fan, Zhiwen and Cong, Wenyan and Wen, Kairun and Wang, Kevin and Zhang, Jian and Ding, Xinghao and Xu, Danfei and Ivanovic, Boris and Pavone, Marco and Pavlakos, Georgios and others},
  journal={arXiv preprint arXiv:2403.20309},
  volume={2},
  year={2024}
}
```
