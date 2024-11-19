import taichi as ti
import numpy as np
import os
import imageio.v2 as imageio 

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

# simulation parameters
dim, n_grid, steps, dt = 3, 32, 25, 4e-4
n_particles = n_grid ** dim // 2 ** (dim - 1)
dx = 1 / n_grid
p_rho = 1
p_vol = (dx * 0.5) ** 3
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3

# variable elastic parameters and viscosity coefficients
E = ti.field(float, (n_grid,)*dim)  
nu = 0.2  
mu = ti.field(float, (n_grid,)*dim) 
lam = ti.field(float, (n_grid,)*dim) 

# particle and grid fields
F_x = ti.Vector.field(dim, float, n_particles)  # pos
F_v = ti.Vector.field(dim, float, n_particles)  # v
F_C = ti.Matrix.field(dim, dim, float, n_particles)  # deformation gradient
F_J = ti.field(float, n_particles)  # volume change rate
F_density = ti.field(dtype=ti.f32, shape=n_particles)  # Density
F_pressure = ti.field(dtype=ti.f32, shape=n_particles)  # pressure
F_cov = ti.Matrix.field(dim, dim, float, n_particles)  # Covariance matrix
F_init_cov = ti.Matrix.field(dim, dim, float, n_particles)  

F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)  # grid speed
F_grid_m = ti.field(float, (n_grid,) * dim) # mesh quality

neighbour = (3,) * dim

# SPH kernel function
@ti.func
def sph_kernel(r, h):
    q = r / h
    value = 0.0
    if q < 1.0:
        value = (1.0 - q)**3
    return value


@ti.kernel
def initialize_fields():
    for I in ti.grouped(E):
        E[I] = 100 + ti.random() * 300 if I[0] < n_grid // 2 else 300 + ti.random() * 300
        mu[I] = E[I] / (2 * (1 + nu))
        lam[I] = E[I] * nu / ((1 + nu) * (1 - 2 * nu))

    for i in range(n_particles):
        F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.4 + 0.15
        F_v[i] = ti.Vector([0.0 for _ in range(dim)])
        F_C[i] = ti.Matrix.zero(float, dim, dim)
        F_J[i] = 1.0
        F_cov[i] = ti.Matrix.identity(float, dim) * 0.01  # 初始化协方差
        F_init_cov[i] = F_cov[i]


@ti.kernel
def compute_density_and_pressure():
    """Calculate particle density and pressure"""
    h = 0.1  # smoothing radius
    for i in range(n_particles):
        F_density[i] = 0.0
        for j in range(n_particles):
            r = (F_x[i] - F_x[j]).norm()
            F_density[i] += p_mass * sph_kernel(r, h)
        # equation state
        F_pressure[i] = ti.max(0.0, 1.0 * (F_density[i] - 1000.0))  


@ti.kernel
def update_cov():
    """update the covariance matrix based on the deformation gradient"""
    for i in range(n_particles):
        cov_n = F_cov[i]  
        grad_v = F_C[i]  

        # update the covariance matrix using the formula
        cov_np1 = cov_n + dt * (grad_v @ cov_n + cov_n @ grad_v.transpose())

        for d in ti.static(range(dim)):
            cov_np1[d, d] = ti.max(cov_np1[d, d], 0.01)
        # symmetric processing
        F_cov[i] = (cov_np1 + cov_np1.transpose()) * 0.5


@ti.kernel
def compute_cov_from_F():
    
    for i in range(n_particles):
        F = F_C[i]  
        init_cov = F_init_cov[i] 
        # new cov matrix
        cov = F @ init_cov @ F.transpose()
        F_cov[i] = cov
        
@ti.func
def matrix_inverse_3x3(A):
    """Compute the inverse of a 3x3 matrix A."""
    det = A.determinant()
    assert det != 0, "matrix is singular, cannot be inverted."
    adj = ti.Matrix([
        [A[1,1] * A[2,2] - A[1,2] * A[2,1], A[0,2] * A[2,1] - A[0,1] * A[2,2], A[0,1] * A[1,2] - A[0,2] * A[1,1]],
        [A[1,2] * A[2,0] - A[1,0] * A[2,2], A[0,0] * A[2,2] - A[0,2] * A[2,0], A[0,2] * A[1,0] - A[0,0] * A[1,2]],
        [A[1,0] * A[2,1] - A[1,1] * A[2,0], A[0,1] * A[2,0] - A[0,0] * A[2,1], A[0,0] * A[1, 1] - A[0,1] * A[1,0]]
    ])
    return adj / det

@ti.func
def polar_decomposition(F, max_iter=10, tol=1e-6):
    """Perform polar decomposition on the deformation gradient F using an iterative method."""
    R = F
    for _ in range(max_iter):
        R_inv_transpose = matrix_inverse_3x3(R.transpose())  # Replace ti.inv
        R_next = 0.5 * (R + R_inv_transpose)
        if (R_next - R).norm() < tol:  # Check for convergence
            break
        R = R_next
    return R

@ti.kernel
def substep():
    """Main time-stepping function."""
    # Reset the grid
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0.0

    # P2G
    for p in range(n_particles):
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E[base] * (F_J[p] - 1) * p_vol / dx ** 2
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
        if F_grid_m[I]> 0:
            # convert momentum to velocity
            F_grid_v[I] /= F_grid_m[I]  
        F_grid_v[I][1] -= dt*gravity  
        # boundary conditions
        
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

        # apply polar decomposition to F_C
        R = polar_decomposition(F_C[p])  # extract rotation matrix
        # R can be further used here if needed



def render(gui, frame, output_dir):
    pos = F_x.to_numpy()[:, :2] 
    pos = np.clip(pos, 0, 1)  
    density = F_density.to_numpy()
    
    min_density, max_density = np.percentile(density, [5,95]) 
    density_normalized = np.clip((density-min_density) / (max_density-min_density), 0,1)
    
    colors = (density_normalized*255).astype(np.uint32)
    gui.circles(pos, radius=1.5, color=colors * 0x0000FF + (255 - colors) * 0x99CCFF)
    gui.show(f"{output_dir}/frame_{frame:04d}.png")


def main():
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)
    
    initialize_fields()
    gui = ti.GUI("MPM Simulation with SPH and Visual Enhancements", res=(800, 800))
    for frame in range(500):
        compute_density_and_pressure() 
        update_cov()  
        compute_cov_from_F()  
        for _ in range(steps):
            substep()
        render(gui, frame, output_dir)
    
    frames = [imageio.imread(f"{output_dir}/frame_{i:04d}.png") for i in range(500)]
    imageio.mimsave("simulation.mp4", frames, fps=30)


if __name__ == "__main__":
    main()
