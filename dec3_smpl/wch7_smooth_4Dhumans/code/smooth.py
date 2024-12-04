import pickle
import numpy as np
import pdb
from scipy.spatial.transform import Rotation as R

'''
    simple script to smooth the human pose from the output of humans 4D pkl result
    largely refers to the github issue discussion (https://github.com/shubham-goel/4D-Humans/issues/33)
    and the code here (https://github.com/haofanwang/CLIFF/blob/main/demo.py#L354-L407)
    need to install the mmhuman3d package becuase we want to use the smoothnet
'''

# define the pkl output file paths and load them
file = "outputs/results/demo_wrestling_I.pkl"
file_smooth = "outputs/results/demo_wrestling_I_smooth.pkl"
import joblib
output = joblib.load(file)

# loop through the frames and characters to get the pose, however, for the sake of time
# not able to handle missing characters, better that all the humans are detected in all the frames
num_character_in_videos = 5
for num_character in range(num_character_in_videos):
    for fframe, data in enumerate(output.items()):
        # note that this should be camera_bbox, not camera, see the discussion in the github issue
        trans = data[1]['camera_bbox'][num_character]
        global_orient = data[1]['smpl'][num_character]['global_orient']
        body_pose = data[1]['smpl'][num_character]['body_pose']
        final_body_pose = np.vstack([global_orient, body_pose])

        r = R.from_matrix(final_body_pose)
        body_pose_vec = r.as_rotvec() 
        trans_temp = [trans[0],trans[1],0]
        
        # following here (https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf)
        # smpl full pose in axis-angle is 72 dimensional
        if fframe==0:
            smpl_trans =trans_temp
            smpl_pose = body_pose_vec.reshape(1,72)
        else:
            smpl_pose = np.vstack([smpl_pose,body_pose_vec.reshape(1,72)])
            smpl_trans = np.vstack([smpl_trans,trans_temp])

    # check the dimensions of the pose 
    print('trans: ',smpl_trans.shape) # (N,72)
    print('shape: ',smpl_pose.shape) # (N,3)

    
    # start smoothing process
    import copy
    import numpy as np
    from mmhuman3d.utils.demo_utils import smooth_process

    pose = smpl_pose
    trans = smpl_trans
    # can change to different smooth window size here
    smooth_type = 'smoothnet_windowsize16'
    
    # start from 0, the interval is 2
    p0 = pose[::2]
    t0 = trans[::2]
    frame_num = p0.shape[0]
    print(frame_num)
    new_pose_0 = smooth_process(p0.reshape(frame_num,24,3), 
                                # smooth_type='smoothnet_windowsize8',
                                smooth_type=smooth_type,
                                cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,72)

    new_trans_0 = smooth_process(t0[:, np.newaxis], 
                                    # smooth_type='smoothnet_windowsize8',
                                    smooth_type=smooth_type,
                                    cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,3)

    # starting from here, refers to (https://github.com/haofanwang/CLIFF/blob/main/demo.py#L354-L407)
    # start from 1, the interval is 2
    # remember to get the config for smoothed net
    p1 = pose[1::2]
    t1 = trans[1::2]
    frame_num = p1.shape[0]
    new_pose_1 = smooth_process(p1.reshape(frame_num,24,3), 
                                # smooth_type='smoothnet_windowsize8',
                                smooth_type=smooth_type_s,
                                cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,72)

    new_trans_1 = smooth_process(t1[:, np.newaxis], 
                                    # smooth_type='smoothnet_windowsize8',
                                    smooth_type=smooth_type_s,
                                    cfg_base_dir='configs/_base_/post_processing/').reshape(frame_num,3)
    new_pose = copy.copy(pose)
    new_trans = copy.copy(trans)
    new_pose[::2] = new_pose_0
    new_pose[1::2] = new_pose_1
    new_trans[::2] = new_trans_0
    new_trans[1::2] = new_trans_1

    # create the smoothed pickle file
    for ffnew, data_new in enumerate(output.items()):
        ## convert rotation vector back into rotation matrix
        r_new = R.from_rotvec(new_pose[ffnew].reshape(24,3))
        body_pose_matrix = r_new.as_matrix() 

        global_orient_new = body_pose_matrix[0].reshape(1,3,3)
        body_pose_new = body_pose_matrix[1:]

        # still, use the camera_bbox
        output[data_new[0]]['camera_bbox'][num_character] = new_trans[ffnew]
        output[data_new[0]]['smpl'][num_character]['global_orient'] = global_orient_new
        output[data_new[0]]['smpl'][num_character]['body_pose'] = body_pose_new


with open(file_smooth, 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    