# Smooth 4D Humans
## Task description
[4DHumans](https://shubham-goel.github.io/4dhumans/) is able to   reconstruct humans and track humans over time with an input video. Though the reconstructions are temporally stable compared to the baselines, we still observe obvious jittering in videos like wrestling. This file folder contains the hacker code to try to smooth the result given by Humans4D by post processing the smpl pose it gives.

## Set Up
1. Follow the [4DHumans guidline](https://github.com/shubham-goel/4D-Humans) to setup the needed environment and configurations, in addition, one need to install ```mmcv``` and ```mmhuman3d``` to use [SmoothNet](https://ailingzeng.site/smoothnet) as smoothing technique.
2. copy ```code/smooth.py``` and ```code/render.py``` to the main folder of 4DHumans repo, and copy ```configs``` folder (for SmoothNet) to the main folder.
3. if you want to reproduce the wrestling video, you can also copy the videos in ```example_data/videos``` into ```example_data/videos``` in the 4DHumans repo, you can also download and put any video you want to try there

## Run the demo
After copying the code and video materials to 4DHumans repo. First follow [4DHumans guidline](https://github.com/shubham-goel/4D-Humans), run the video track for the video you have, e.g.:
```
python track.py video.source="example_data/videos/wrestling_I.mp4"
``` 
This will output the ```.pkl``` result in ```output/results``` folder. Then run the smoothing code (need to check the input result and smoothed output path which is assigned inside the code)
```
python smooth.py
```
Then check the smooth result by rendering it back to the video (also need to check the ```.pkl``` file path coded in side)
```
python render.py
```

## Results
The results (two wrestling video and slow motion GIFs) are shown in the  ```result_comparisons``` folder
## References
The code of smoothing largely refers to:
[this discussioin](https://github.com/shubham-goel/4D-Humans/issues/33) and this [repo](https://github.com/haofanwang/CLIFF/blob/main/demo.py) and the [smpl made simple F&Q](https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf)

The render function used to render  ```.pkl``` result back into video is actually a simplified reformulation of the ```PHALP.track()``` function in [PHALP repo](https://github.com/brjathu/PHALP/blob/master/phalp/trackers/PHALP.py#L179-L266) 