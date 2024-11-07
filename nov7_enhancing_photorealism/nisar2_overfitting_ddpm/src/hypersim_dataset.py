import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class HypersimDataset(Dataset):
    def __init__(self, root_dir, scenes_cams_frames_pth, transform=None):
        self.root_dir = root_dir
        self.scenes_cams_frames = pd.read_csv(scenes_cams_frames_pth, dtype={'frame': str})
        self.transform = transform

    def __len__(self):
        return len(self.scenes_cams_frames)

    def __getitem__(self, idx):
        scene_cam_frame = self.scenes_cams_frames.iloc[idx]
        
        scene = scene_cam_frame.loc['scene']
        cam = scene_cam_frame.loc['cam']
        frame = scene_cam_frame.loc['frame']
        images_pth = os.path.join(self.root_dir, scene, 'images')

        final_preview_pth = os.path.join(images_pth, f'scene_{cam}_final_preview')

        final_color_pth = os.path.join(final_preview_pth, f'frame.{frame}.color.jpg')
        final_color = Image.open(final_color_pth)
        if self.transform:
            final_color = self.transform(final_color)

        final_diff_pth = os.path.join(final_preview_pth, f'frame.{frame}.diff.jpg')
        final_diff = Image.open(final_diff_pth)
        if self.transform:
            final_diff = self.transform(final_diff)

        final_diffuse_illumination_pth = os.path.join(final_preview_pth, f'frame.{frame}.diffuse_illumination.jpg')
        final_diffuse_illumination = Image.open(final_diffuse_illumination_pth)
        if self.transform:
            final_diffuse_illumination = self.transform(final_diffuse_illumination)

        final_diffuse_reflectance_pth = os.path.join(final_preview_pth, f'frame.{frame}.diffuse_reflectance.jpg')
        final_diffuse_reflectance = Image.open(final_diffuse_reflectance_pth)
        if self.transform:
            final_diffuse_reflectance = self.transform(final_diffuse_reflectance)

        final_gamma_pth = os.path.join(final_preview_pth, f'frame.{frame}.gamma.jpg')
        final_gamma = Image.open(final_gamma_pth)
        if self.transform:
            final_gamma = self.transform(final_gamma)

        final_lambertian_pth = os.path.join(final_preview_pth, f'frame.{frame}.lambertian.jpg')
        final_lambertian = Image.open(final_lambertian_pth)
        if self.transform:
            final_lambertian = self.transform(final_lambertian)

        final_non_lambertian_pth = os.path.join(final_preview_pth, f'frame.{frame}.non_lambertian.jpg')
        final_non_lambertian = Image.open(final_non_lambertian_pth)
        if self.transform:
            final_non_lambertian = self.transform(final_non_lambertian)

        final_residual_pth = os.path.join(final_preview_pth, f'frame.{frame}.residual.jpg')
        final_residual = Image.open(final_residual_pth)
        if self.transform:
            final_residual = self.transform(final_residual)

        final_tonemap_pth = os.path.join(final_preview_pth, f'frame.{frame}.tonemap.jpg')
        final_tonemap = Image.open(final_tonemap_pth)
        if self.transform:
            final_tonemap = self.transform(final_tonemap)


        geo_preview_pth = os.path.join(images_pth, f'scene_{cam}_geometry_preview')

        geo_preview_color_pth = os.path.join(geo_preview_pth, f'frame.{frame}.color.jpg')
        geo_preview_color = Image.open(geo_preview_color_pth)
        if self.transform:
            geo_preview_color = self.transform(geo_preview_color)

        geo_depth_meter_pth = os.path.join(geo_preview_pth, f'frame.{frame}.depth_meters.png')
        geo_depth_meter = Image.open(geo_depth_meter_pth)
        if self.transform:
            geo_depth_meter = self.transform(geo_depth_meter)[:3,:,:]

        geo_gamma_pth = os.path.join(geo_preview_pth, f'frame.{frame}.gamma.jpg')
        geo_gamma = Image.open(geo_gamma_pth)
        if self.transform:
            geo_gamma = self.transform(geo_gamma)

        geo_normal_bump_cam_pth = os.path.join(geo_preview_pth, f'frame.{frame}.normal_bump_cam.png')
        geo_normal_bump_cam = Image.open(geo_normal_bump_cam_pth)
        if self.transform:
            geo_normal_bump_cam = self.transform(geo_normal_bump_cam)[:3,:,:]

        geo_normal_bump_world_pth = os.path.join(geo_preview_pth, f'frame.{frame}.normal_bump_world.png')
        geo_normal_bump_world = Image.open(geo_normal_bump_world_pth)
        if self.transform:
            geo_normal_bump_world = self.transform(geo_normal_bump_world)[:3,:,:]

        geo_normal_cam_pth = os.path.join(geo_preview_pth, f'frame.{frame}.normal_cam.png')
        geo_normal_cam = Image.open(geo_normal_cam_pth)
        if self.transform:
            geo_normal_cam = self.transform(geo_normal_cam)[:3,:,:]

        geo_render_entity_id_pth = os.path.join(geo_preview_pth, f'frame.{frame}.render_entity_id.png')
        geo_render_entity_id = Image.open(geo_render_entity_id_pth)
        if self.transform:
            geo_render_entity_id = self.transform(geo_render_entity_id)[:3,:,:]

        geo_semantic_instance_pth = os.path.join(geo_preview_pth, f'frame.{frame}.semantic_instance.png')
        geo_semantic_instance = Image.open(geo_semantic_instance_pth)
        if self.transform:
            geo_semantic_instance = self.transform(geo_semantic_instance)[:3,:,:]

        geo_semantic_pth = os.path.join(geo_preview_pth, f'frame.{frame}.semantic.png')
        geo_semantic = Image.open(geo_semantic_pth)
        if self.transform:
            geo_semantic = self.transform(geo_semantic)[:3,:,:]

        geo_tex_coord_pth = os.path.join(geo_preview_pth, f'frame.{frame}.tex_coord.png')
        geo_tex_coord = Image.open(geo_tex_coord_pth)
        if self.transform:
            geo_tex_coord = self.transform(geo_tex_coord)[:3,:,:]

        return {
            'final_color': final_color,
            # 'final_diff': final_diff,
            'final_diffuse_illumination': final_diffuse_illumination,
            'final_diffuse_reflectance': final_diffuse_reflectance,
            'final_gamma': final_gamma,
            'final_lambertian': final_lambertian,
            'final_non_lambertian': final_non_lambertian,
            # 'final_residual': final_residual,
            'final_tonemap': final_tonemap,
            # 'geo_preview_color': geo_preview_color,
            # 'geo_depth_meter': geo_depth_meter,
            'geo_gamma': geo_gamma,
            # 'geo_normal_bump_cam': geo_normal_bump_cam,
            'geo_normal_bump_world': geo_normal_bump_world,
            # 'geo_normal_cam': geo_normal_cam,
            # 'geo_render_entity_id': geo_render_entity_id,
            # 'geo_semantic_instance': geo_semantic_instance,
            # 'geo_semantic': geo_semantic,
            'geo_tex_coord': geo_tex_coord
        }


    
if __name__ == '__main__':
    root_dir = '/home/nisar2/Documents/cs598_3dv_hacker/hypersim_data'
    scenes_cams_frames_pth = '/home/nisar2/Documents/cs598_3dv_hacker/src/scene_cam_frame.csv'

    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = HypersimDataset(root_dir, scenes_cams_frames_pth, transform=transforms)
    data = dataset[0]
    for key, value in data.items():
        print(f"{key}: {value.shape}")
    print(data['geo_depth_meter'][:3,:,:].shape)
    plt.imshow(data['geo_depth_meter'][:3,:,:].permute(1,2,0))
    plt.show()

