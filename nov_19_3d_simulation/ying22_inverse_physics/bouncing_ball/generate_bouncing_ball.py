import json
import pdb
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu) 

# parameters
dt = 0.01  
gravity = -9.8
bounciness = 0.8

# Canvas size
width, height = 800, 800

# Ball properties
ball_radius = 0.05
position = ti.Vector.field(2, dtype=ti.f32, shape=())
velocity = ti.Vector.field(2, dtype=ti.f32, shape=())

# Video Manager
output_dir = "./bouncing_ball_video"
video_manager = ti.tools.VideoManager(output_dir=output_dir, framerate=60)

# Initial position and velocity
@ti.kernel
def initialize():
    position[None] = ti.Vector([0.5, 0.8])  
    velocity[None] = ti.Vector([0.2, -0.5])
    

# Update ball position and velocity
@ti.kernel
def update():
    velocity[None].y += gravity * dt
    position[None] += velocity[None] * dt

    # Collision with ground
    if position[None].y - ball_radius <= 0.0:
        position[None].y = ball_radius  # Prevent sinking into the ground
        velocity[None].y = -velocity[None].y * bounciness  # Reverse and dampen velocity
    
    # Collision with walls
    if position[None].x - ball_radius <= 0.0 or position[None].x + ball_radius >= 1.0:
        velocity[None].x = -velocity[None].x
    

# Rendering
initialize()
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))  # RGB image

@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])  # Black background

        # Map ball position to screen coordinates
        ball_screen_pos = ti.Vector([position[None].x, position[None].y]) * ti.Vector([width, height])
        ball_screen_pos = ball_screen_pos.cast(ti.i32)
        

        # Render the ball
        dist = ti.sqrt((i - ball_screen_pos.x)**2 + (j - ball_screen_pos.y)**2)
        if dist < ball_radius * width:
            pixels[i, j] = ti.Vector([1.0, 0.8, 0.4])  # Ball color (yellow-orange)

# Define the trajectory
gt_trajectory = []

# Simulation
for frame in range(300):  # Simulate for 300 frames
    update()
    
    gt_trajectory.append((frame * dt, position[None].x, position[None].y))
    
    render()
    video_manager.write_frame(pixels)

# Create video
video_manager.make_video(gif=True)
print(f"Video saved to {output_dir}")

breakpoint()
with open('gt_trajectory.json', 'w') as f:
    json.dump({'gt_trajectory': gt_trajectory}, f)