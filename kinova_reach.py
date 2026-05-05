# Kinova gen3 sphere pushing task
# uses differential IK to control the end effector through a 3-phase push task
# robot model : kinova gen3 7-DOF (no gripper)
# framework: NVIDIA Isaac Lab 2.1.0
# author: Youcef Sellai

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG

# imports for the IK

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


#sim setup

# the simulation context
sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)

# creating the ground

sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

# creating the robot

robot_cfg = KINOVA_GEN3_N7_CFG.replace(prim_path = "/World/Robot")
robot = Articulation(robot_cfg)

# creating a sphere prop

sphere_cfg = RigidObjectCfg(
    prim_path ="/World/sphere",
    spawn = sim_utils.SphereCfg(
        radius = 0.05,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
                linear_damping = 0.1,
                angular_damping = 0.1,
        ),
        mass_props = sim_utils.MassPropertiesCfg(mass = 0.5),
        collision_props = sim_utils.CollisionPropertiesCfg(),
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (1.0, 0.0, 0.0))
    ),
    init_state = RigidObjectCfg.InitialStateCfg(
        pos = (0.45, 0.0, 0.15)
    )
)

sphere = RigidObject(sphere_cfg)

sim.reset()

# setting up the IK controller

# configure the controller

ik_cfg = DifferentialIKControllerCfg(
    command_type = "pose",
    use_relative_mode = False,
    ik_method = "dls",
)

#create the controller

ik_controller = DifferentialIKController(
    cfg = ik_cfg,
    num_envs = 1,
    device = sim.device
)

# telling the controller which body is the end effector

robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[1-7]"], body_names=["end_effector_link"])
robot_entity_cfg.resolve({"robot": robot})
ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

# movement loop

# initial target position in 3d

target_pos = torch.tensor([[0.3, 0.0, 0.3]], device = sim.device)
target_rot = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device = sim.device) # points ee downward

#combining pos and rot into one tensor
target_pose = torch.cat([target_pos, target_rot], dim = 1)

ik_controller.reset()

ik_controller.set_command(target_pose)


for i in range(2500):

    #get current ee state

    ee_pos_w = robot.data.body_pos_w[:, ee_jacobi_idx, :]
    ee_rot_w = robot.data.body_quat_w[:, ee_jacobi_idx, :]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
        ee_rot_w,

    )

    #get jacobian for ik computation

    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

    #computing ik

    joint_targets = ik_controller.compute(
        ee_pos_b,
        ee_quat_b,
        jacobian,
        joint_pos,
    )

    # apply to full robot

    robot.set_joint_position_target(joint_targets, joint_ids = robot_entity_cfg.joint_ids)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())

    #updating the sphere
    sphere.update(sim.get_physics_dt())

    #keeping track

    if i % 100 == 0:
        ee_pos = robot.data.body_pos_w[:, robot_entity_cfg.body_ids[0], :]
        sphere_pos = sphere.data.root_pos_w
        dist = torch.norm(ee_pos - target_pos)
        print(f"Step {i} | EE_X: {ee_pos[0,0].item():.3f} | Sphere_X: {sphere_pos[0,0].item():.3f} | Dist: {dist:.3f}m")

    # phase 2 : descending toward the center of the sphere

    if i ==400:
        print("\nPhase 2 starting : descending toward the sphere..")
        target_pos = torch.tensor([[0.3, 0.0, 0.15]], device = sim.device)
        target_pose = torch.cat([target_pos, target_rot], dim = 1)
        ik_controller.reset()
        ik_controller.set_command(target_pose)

    #phase 3 : pushing the sphere forward
    if i >= 600:
        if i == 600:
            print("\nPhase 3 starting : pushing the sphere forward..")
            ik_controller.reset()
        
        # linear interpolation to push gradually
        progress = (i-600)/300
        progress = min(progress, 1.0) # caps at 1.0 so it stops at target
        current_x = 0.3 + progress * 0.35
        target_pos = torch.tensor([[current_x, 0.0, 0.15]], device = sim.device)
        target_pose = torch.cat([target_pos, target_rot], dim = 1)
        ik_controller.set_command(target_pose)

# success check
final_sphere_x = sphere.data.root_pos_w[0, 0].item()
push_distance = final_sphere_x - 0.45

print(f"\nTask completed")
print(f"Sphere pushed {push_distance:.3f}m forward")

if push_distance > 0.1:
    print("SUCCESS : Sphere successfully pushed")
else:
    print("INCOMPLETE : Sphere barely moved")

simulation_app.close()