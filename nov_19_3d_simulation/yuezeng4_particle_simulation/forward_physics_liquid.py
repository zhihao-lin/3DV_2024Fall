##liquid

import taichi as ti
import numpy as np
import os
import imageio.v2 as imageio

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

# simulation parameters
dim, n_grid, steps, dt = 3, 32, 25, 4e-4
n_particles = n_grid ** dim // 2**(dim - 1)
dx = 1/n_grid
p_rho = 1
p_vol = (dx * 0.5) ** 3
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

# particle and grid fields
F_x = ti.Vector.field(dim, float, n_particles)  
F_v = ti.Vector.field(dim, float, n_particles) 
F_C = ti.Matrix.field(dim, dim, float, n_particles)  
F_J = ti.field(float, n_particles)  

F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m = ti.field(float, (n_grid,) * dim) 


neighbour = (3,) * dim


@ti.kernel
def init():
    """initialize particle position and volume change rate"""
    for i in range(n_particles):
        F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.4 + 0.15
        F_v[i] = ti.Vector([0.0 for _ in range(dim)])
        F_C[i] = ti.Matrix.zero(float, dim, dim)
        F_J[i] = 1.0


@ti.kernel
def substep():
    """time step function"""
    # 重置网格
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0.0

    # 粒子到网格 (P2G)
    for p in range(n_particles):
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * (F_J[p] - 1) * p_vol / dx ** 2
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass

    # grid update
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            # momentum to speed
            F_grid_v[I] /= F_grid_m[I] 
        F_grid_v[I][1] -= dt * gravity 
        # boundary processing
        for d in ti.static(range(dim)):
            if I[d] < bound and F_grid_v[I][d] < 0:
                F_grid_v[I][d] = 0
            if I[d] > n_grid - bound and F_grid_v[I][d] > 0:
                F_grid_v[I][d] = 0

    # G2P
    for p in range(n_particles):
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_J[p] *= 1 + dt * new_C.trace()
        F_C[p] = new_C


@ti.kernel
def validate_positions():
    """Verify whether the particle position is out of bounds"""
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            if F_x[i][d] < 0 or F_x[i][d] > 1:
                print(f"particles {i} on dim {d} out of bounds on, position {F_x[i][d]}")


def render(gui, frame, output_dir):
    """Render particles and save each frame."""
    pos = F_x.to_numpy()[:, :2]  # Project to 2D
    pos = np.clip(pos, 0, 1)  # Clip to valid range
    gui.circles(pos, radius=1.5, color=0x66CCFF)
    gui.show(f"{output_dir}/frame_{frame:04d}.png")


def save_to_mp4(output_dir, video_filename, num_frames, fps=30):
    """save simulation frames as an MP4 video."""
    frames = []
    for i in range(num_frames):
        frame_path = f"{output_dir}/frame_{i:04d}.png"
        frames.append(imageio.imread(frame_path))
    imageio.mimsave(video_filename, frames, fps=fps)


def main():
    output_dir = "frames"
    video_filename = "simulation_liquid.mp4"
    os.makedirs(output_dir, exist_ok=True)

    init()
    gui = ti.GUI("MPM Simulation", res=(800, 800))
    num_frames = 500

    for frame in range(num_frames):
        for _ in range(steps):
            substep()
        render(gui, frame, output_dir)

    save_to_mp4(output_dir, video_filename, num_frames, fps=30)


if __name__ == "__main__":
    main()






