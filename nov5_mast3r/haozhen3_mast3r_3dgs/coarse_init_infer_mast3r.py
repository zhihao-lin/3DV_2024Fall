import sys
import os 
import torch
import time
import copy
import numpy as np
import argparse

def get_intrinsics(focals, principal_points, device='cuda:0'):
    K = torch.zeros((len(focals), 3, 3), device=device)
    for i in range(len(focals)):
        K[i, 0, 0] = K[i, 1, 1] = focals[i]
        K[i, :2, 2] = principal_points[i]
        K[i, 2, 2] = 1
    return K

def parse_args():
    parser = argparse.ArgumentParser(description="MASt3R Point Cloud Optimization")
    parser.add_argument("--device", type=str, default='cuda:0', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument('--lr1', type=float, default=0.07, help="Learning rate for coarse global alignment")
    parser.add_argument('--lr2', type=float, default=0.014, help="Learning rate for refinement")
    parser.add_argument('--niter1', type=int, default=800, help="Number of iterations for coarse global alignment")
    parser.add_argument('--niter2', type=int, default=400, help="Number of iterations for refinement")
    parser.add_argument('--optim_level', type=str, default='coarse', choices=['coarse', 'refine', 'refine+depth'], help="Optimization level")
    parser.add_argument('--matching_conf_thr', type=float, default=5, help="Matching confidence threshold")
    parser.add_argument('--shared_intrinsics', type=bool, default=False, help="Whether to use shared intrinsics")
    parser.add_argument('--cache_dir', type=str, default='/data/haozhen/gaussian-model/InstantSplat/output/cache', 
                        help="Path to the cache directory")
    parser.add_argument('--clean_depth', type=bool, default=False, help="Whether to clean the depth maps")
    parser.add_argument('--min_conf_thr', type=float, default=1.5, help="Minimum confidence threshold")
    parser.add_argument('--subsample', type=int, default=8, help="Subsample factor for depth maps")
    parser.add_argument('--img_base_path', type=str, default='/data/haozhen/gaussian-model/InstantSplat/data/TT/Family/6_views/images',
                        help="Path to the image folder")
    parser.add_argument("--model_path", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", help="path to the model weights")
    parser.add_argument("--n_views", type=int, default=6)

    parser.add_argument('--dust3r_path', type=str, 
                        default=os.path.abspath("/data/haozhen/gaussian-model/InstantSplat/submodules/mast3r/dust3r"),
                        help="Path to the Dust3r module")
    parser.add_argument('--mast3r_path', type=str, 
                        default=os.path.abspath("/data/haozhen/gaussian-model/InstantSplat/submodules/mast3r"),
                        help="Path to the Dust3r module")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    sys.path.append('submodule')
    mast3r_path = os.path.abspath(args.mast3r_path)
    sys.path.insert(0, mast3r_path)
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import mast3r.utils.path_to_dust3r
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

    dust3r_path = os.path.abspath(args.dust3r_path)
    sys.path.insert(0, dust3r_path)
    import dust3r
    print(dust3r.__file__)
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.device import to_numpy
    from utils.dust3r_utils import  compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images

    device = args.device
    n_views = args.n_views
    lr1 = args.lr1
    lr2 = args.lr2
    niter1 = args.niter1
    niter2 = args.niter2
    optim_level = args.optim_level
    matching_conf_thr = args.matching_conf_thr
    shared_intrinsics = args.shared_intrinsics
    cache_dir = args.cache_dir
    clean_depth = args.clean_depth
    min_conf_thr = args.min_conf_thr
    subsample = args.subsample
    model_name = args.model_path
    img_folder_path = os.path.join(args.img_base_path, 'images')
    output_colmap_path=img_folder_path.replace("images", "sparse/0")
    train_img_list = sorted(os.listdir(img_folder_path))

    assert len(train_img_list)==n_views, f"Number of images ({len(train_img_list)}) in the folder ({img_folder_path}) is not equal to {n_views}"
    imgs, ori_size = load_images(img_folder_path, size=args.image_size)
    print("ori_size", ori_size)

    # begin
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    print('finish loading model.')
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        train_img_list = [train_img_list[0], train_img_list[0] + '_2']
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

    print(f'Inference with model on {len(pairs)} image pairs')
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=False)
    scene = sparse_global_alignment(train_img_list, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, subsample=subsample)

    
    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    # pts3d = to_numpy(scene.get_sparse_pts3d())
    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    pts3d = [pts.reshape(imgs[0].shape) for pts in pts3d]
    confidence_masks = to_numpy([c > min_conf_thr for c in confs])
    focals = scene.get_focals()
    principal_points = scene.get_principal_points()
    intrinsics = to_numpy(get_intrinsics(focals, principal_points, device))

    # print(imgs[0].shape, focals.shape, poses.shape, pts3d[0].shape, confidence_masks[0].shape, intrinsics[0].shape)
    
    # save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", np.array(focals.cpu()))


