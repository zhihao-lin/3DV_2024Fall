import json
import pdb
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

np.set_printoptions(precision=4, suppress=True)

# Ball radius and initial position
radius = np.float32(0.05)
center_x, center_y = np.float32(0.5), np.float32(0.8)

# Load observed trajectory
with open('gt_trajectory.json', 'r') as f:
    traj_data = json.load(f)["gt_trajectory"]

times, x_obs, y_obs = [], [], []
for traj in traj_data:
    times.append(np.float32(traj[0]))
    x_obs.append(np.float32(traj[1]))
    y_obs.append(np.float32(traj[2]))

times = np.array(times, dtype=np.float32)
x_obs = np.array(x_obs, dtype=np.float32)
y_obs = np.array(y_obs, dtype=np.float32)

def simulate_trajectory(params, times, initial_position, ball_radius):
    """
    Simulates the trajectory of a bouncing ball based on given parameters.

    Returns:
    - x_traj: Array of x positions over time.
    - y_traj: Array of y positions over time.
    """
    gravity, vx0, vy0, bounciness = [np.float32(p) for p in params]
    dt = np.float32(times[1] - times[0])  # assumed constant
    
    # Initialize position and velocity
    x, y = [np.float32(p) for p in initial_position]
    vx, vy = np.float32(vx0), np.float32(vy0)

    x_traj = []
    y_traj = []

    for _ in times:
        vy += gravity * dt

        x += vx * dt
        y += vy * dt

        # Ground collision
        if y - ball_radius <= 0:
            y = ball_radius
            vy = -vy * np.float32(bounciness)  

        # Wall collisions (walls at x=0 and x=1)
        if x - ball_radius <= 0 or x + ball_radius >= 1.0:
            vx = -vx 
        
        x_traj.append(x)
        y_traj.append(y)

    return np.array(x_traj, dtype=np.float32), np.array(y_traj, dtype=np.float32)

def cost_function(params, times, x_obs, y_obs, initial_position, ball_radius):
    """
    Computes the error between observed and simulated trajectories.
    """
    x_sim, y_sim = simulate_trajectory(params, times, initial_position, ball_radius)
    error = np.sum(((x_obs - x_sim) ** 2 + (y_obs - y_sim) ** 2).astype(np.float32))
    return np.float32(error)


def compute_gradient(params, times, x_obs, y_obs, initial_position, ball_radius, epsilon=1e-4):
    grad = np.zeros_like(params, dtype=np.float32)
    for i in range(len(params)):
        params[i] += epsilon
        cost_plus = cost_function(params, times, x_obs, y_obs, initial_position, ball_radius)
        params[i] -= 2 * epsilon
        cost_minus = cost_function(params, times, x_obs, y_obs, initial_position, ball_radius)
        grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
        params[i] += epsilon  
    return grad

def compute_bounciness_from_heights(y_obs, times):
    """
    Compute the bounciness (coefficient of restitution) from observed bounce heights.

    Returns:
    - bounciness: Estimated coefficient of restitution (e).
    - bounce_heights: List of detected bounce heights.
    """
    # Find local maxima (peaks) in the observed trajectory
    peaks, _ = find_peaks(y_obs)
    bounce_heights = np.concatenate([np.array([y_obs[0]]), y_obs[peaks]])

    # Compute bounciness from consecutive bounce heights
    bounciness_list = []
    for i in range(len(bounce_heights) - 1):
        h_before = bounce_heights[i]
        h_after = bounce_heights[i + 1]
        if h_before > 0:  # Avoid division by zero
            e = np.sqrt(h_after / h_before)
            bounciness_list.append(e)
    
    # use the first computed bounciness
    bounciness = bounciness_list[0]
    return bounciness, bounce_heights


def gradient_descent(initial_params, times, x_obs, y_obs, initial_position, ball_radius, 
                     learning_rate=1e-2, max_iter=500, tol=1e-6):
    params = np.array(initial_params, dtype=np.float32)
    cost_history = []

    for iteration in range(max_iter):
        # Compute the cost and gradient
        cost = cost_function(params, times, x_obs, y_obs, initial_position, ball_radius)
        grad = compute_gradient(params, times, x_obs, y_obs, initial_position, ball_radius)

        # Update the parameters
        params -= learning_rate * grad
        cost_history.append(cost)

        # Check for convergence
        if np.linalg.norm(grad) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        if iteration % 10 == 0:
            print(f"Iteration {iteration + 1}: Cost = {cost:.6f}, Parameters = {params}")

    return params, cost_history

bounciness, bounce_heights = compute_bounciness_from_heights(y_obs, times)

# Initial guesses for optimization
initial_gravity = -9.8
initial_vx = (x_obs[1] - x_obs[0]) / (times[1] - times[0])
initial_vy = (y_obs[1] - y_obs[0]) / (times[1] - times[0]) - (times[1] - times[0]) * initial_gravity

initial_guess = np.array([initial_gravity, initial_vx, initial_vy, bounciness], dtype=np.float32)

# Start optimization
optimized_params, cost_history = gradient_descent(
    initial_guess, times, x_obs, y_obs, [center_x, center_y], radius, 
    learning_rate=1e-5, max_iter=5000, tol=1e-6
)

print("\nOptimization Results:")
print("Optimized Parameters:")
print(f"Gravity: {optimized_params[0]:.2f}")
print(f"Initial velocity (vx0, vy0): ({optimized_params[1]:.2f}, {optimized_params[2]:.2f})")
print(f"Bounciness: {optimized_params[3]:.2f}")

# Simulate trajectory with optimized parameters
x_sim, y_sim = simulate_trajectory(optimized_params, times, [center_x, center_y], radius)

# Plot loss trajectory
plt.figure(figsize=(10, 6))
plt.plot(cost_history, label="Loss")
plt.title("Loss (Gradient Descent, lr=1e-5)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig("bouncing_ball_loss.png")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_obs, y_obs, 'ro', label="Ground-Truth Trajectory", markersize=3)
plt.plot(x_sim, y_sim, 'b-', label="Fitted Trajectory")
plt.title("Bouncing Ball: Ground-Truth vs Fitted Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.savefig("bouncing_ball_trajectory.png")
