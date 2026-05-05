# Kinova Gen3 Sphere Pushing Task - NVIDIA Isaac Lab

A  robotics simulation demonstrating the Kinova Gen3 7-DOF arm performing a 3-phase sphere pushing task using Differential Inverse Kinematics in NVIDIA Isaac Lab.

## Task Description

The robot executes a 3-phase manipulation task:

- **Phase 1** - Arm reaches a position behind the target sphere
- **Phase 2** - Arm descends to sphere height
- **Phase 3** - Arm smoothly pushes the sphere forward using linear interpolation

The sphere is pushed over 1.5 meters forward from its starting position.

## Technical Details

- **Robot:** Kinova Gen3 7-DOF arm
- **Framework:** NVIDIA Isaac Lab 2.1.0
- **Physics Engine:** NVIDIA PhysX
- **IK Method:** Differential IK with Damped Least Squares (DLS)
- **Trajectory Planning:** Linear interpolation for smooth end-effector motion

## Requirements

- NVIDIA Isaac Lab 2.1.0
- Isaac Sim 4.5.0
- Python 3.10
- CUDA-compatible GPU

## How To Run

```bash
conda activate isaacenv
python kinova_reach.py --headless
```
Remove `--headless` to visualize the simulation in the Isaac Sim GUI.

## What This Demonstrates

- Differential IK control for precise end-effector positioning
- Multi-phase task execution using a state machine
- Physics-based object interaction with a rigid body sphere
- Base frame transformation for correct IK computation
- Linear interpolation for smooth trajectory execution

## Author

Youcef Sellai - Mechanical Engineering Student
Gina Cody Engineering School, Concordia University