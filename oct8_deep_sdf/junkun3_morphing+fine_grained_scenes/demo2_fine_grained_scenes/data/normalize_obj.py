import sys
import numpy as np

def normalize_obj(input_file, output_file):
    vertices = []
    vertex_indices = []

    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    
    from tqdm import tqdm
    for idx, line in enumerate(tqdm(lines)):
        if line.startswith('v '):  
            parts = line.strip().split()
            vertex = [float(coord) for coord in parts[1:4]]
            vertices.append(vertex)
            vertex_indices.append(idx)  

    
    vertices = np.array(vertices)

    
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (max_coords + min_coords) / 2
    scale = (max_coords - min_coords).max() / 2
    normalized_vertices = (vertices - center) / scale

    vertex_indices = set(vertex_indices)

    
    with open(output_file, 'w') as f:
        vertex_counter = 0  
        for idx, line in enumerate(tqdm(lines)):
            if idx in vertex_indices:
                
                vertex = normalized_vertices[vertex_counter]
                
                f.write('v {:.6f} {:.6f} {:.6f}\n'.format(*vertex))
                vertex_counter += 1
            else:
                
                f.write(line)

    print(f'Normalized {output_file}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('python normalize_obj.py .obj')
    else:
        input_file = sys.argv[1]
        import os
        os.system(f'cp {input_file} {input_file}.bak')
        normalize_obj(input_file, input_file)
