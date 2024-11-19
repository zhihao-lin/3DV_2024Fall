# Particle Simulation with Taichi
Hacker: Yue Zeng (yuezeng4@illinois.edu)
This repository contains two particle simulation scripts built using **Taichi**:
1. **`forward_physics_granular.py`**: Simulates granular material behaviors such as sand or soil.
2. **`forward_physics_liquid.py`**: Simulates liquid particle dynamics.

---

## Setup

### System Requirements
- **GPU**: CUDA-enabled GPU recommended (e.g., NVIDIA RTX series).
- **CPU**: A quad-core CPU or higher for CPU-only execution.
- **Memory**: 8GB+ RAM.
- **Operating System**: Linux, macOS, or Windows.

### Python Environment Setup
Follow these steps to set up the Python environment:

1. Create and activate a virtual environment:
   ```bash
   conda create -n taichi_sim python=3.9 -y
   conda activate taichi_sim
