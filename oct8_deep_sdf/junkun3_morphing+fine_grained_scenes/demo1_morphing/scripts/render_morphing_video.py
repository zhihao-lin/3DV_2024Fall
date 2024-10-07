import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_obj_sequence_matplotlib(obj_folder, output_video, fps=12, rotation_axis='y', rotation_speed=2, repeat=4):
    
    obj_files = sorted([f for f in os.listdir(obj_folder) if f.endswith('.obj')])

    frames = []
    axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    rotation_axis_vector = axis_map.get(rotation_axis.lower(), [0, 1, 0])

    
    dpi = 90
    width, height = 800, 600

    from tqdm import tqdm

    def obj_filename_key(obj_file):
        obj_file = obj_file[:-len('.obj')]
        obj_file = obj_file.split('_')
        obj_file[-1] = float(obj_file[-1])
        obj_file[-2] = int(obj_file[-2])
        return obj_file
    obj_files = sorted(obj_files, key=obj_filename_key) * repeat

    def save_video():

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for frame in (frames):
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
        print(f"Saved to {output_video}")

    for idx, obj_file in enumerate(obj_files):
        print(f"Rendering frame {idx+1}/{len(obj_files)}: {obj_file}")

        
        mesh = trimesh.load_mesh(os.path.join(obj_folder, obj_file))
        mesh.apply_scale(1 / mesh.scale)  

        
        face_normals = mesh.face_normals
        
        face_colors = (face_normals + 1) / 2

        
        angle = rotation_speed * idx
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.deg2rad(angle), rotation_axis_vector)
        mesh.apply_transform(rotation_matrix)

        
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')

        faces = mesh.faces
        vertices = mesh.vertices
        mesh_collection = Poly3DCollection(vertices[faces], facecolors=face_colors, linewidths=0.1, edgecolors='k')
        ax.add_collection3d(mesh_collection)

        
        ax.view_init(elev=30, azim=45)
        

        
        scale = mesh.bounding_box.extents.max()
        ax.set_xlim(-scale / 2, scale / 2)
        ax.set_ylim(-scale / 2, scale / 2)
        ax.set_zlim(-scale / 2, scale / 2)

        
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        
        ax.set_proj_type('persp')

        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape((int(height), int(width), 3))
        frames.append(image)

        plt.close(fig)

        if idx % 16 == 0:
            save_video()

    save_video()
    

if __name__ == "__main__":
    obj_folder = './results/runs_sdf/17_07_172540/meshes_training'  
    output_video = './results/morphing_video.mp4'      
    render_obj_sequence_matplotlib(obj_folder, output_video, rotation_axis='z', rotation_speed=5)
