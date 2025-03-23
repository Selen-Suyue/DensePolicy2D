import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist
import torchvision.transforms as T

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import DSP
from eval_agent import Agent
from utils.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform, xyz_rot_transform

default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "num_action": 20,
    "num_inference_step": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": False,
    "discretize_rotation": True,
    "ensemble_mode": "new"
})


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
    mask = (x_mask & y_mask & z_mask)
    points = points[mask]
    colors = colors[mask]
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
    return cloud_final

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def create_input(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    colors, depths => batch coords, batch feats
    """
    cloud = create_point_cloud(colors, depths, cam_intrinsics, voxel_size = voxel_size)
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def unnormalize_action_noproject(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (SAFE_WORKSPACE_MAX - SAFE_WORKSPACE_MIN) + SAFE_WORKSPACE_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # aloha
    if args.aloha:
        print("ACT")

    # no project
    if args.no_project:
        print("NO PROJECT")

    # center crop
    if args.center_crop:
        print("CENTER CROP")
        center_crop = T.Compose([
            T.CenterCrop((720, 720)),
            T.Resize((200, 256)), #  TODO???
        ])

    # policy
    print("Loading policy ...")
    
    policy = DSP(
            num_action = args.num_action,
            input_dim = 6,
            obs_feature_dim = args.obs_feature_dim,
            action_dim = 10,
            hidden_dim = args.hidden_dim,
            nheads = args.nheads,
            num_encoder_layers = args.num_encoder_layers, # 4
            num_decoder_layers = args.num_decoder_layers, # 7
            dropout = args.dropout,
            enable_mba = args.enable_mba,
            obj_dim = args.obj_dim,
        ).to(device)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # evaluation
    agent = Agent(
        robot_ip = "192.168.2.100",
        pc_ip = "192.168.2.35",
        gripper_port = "/dev/ttyUSB0",
        camera_serial = "104122060902",
        camera_serial_hand = "104422070044",
    )
    if not args.no_project:
        if args.projector_mocap_mode:
            print("Projector MOCAP mode")
        projector = Projector(args.calib, mocap_mode=args.projector_mocap_mode) # Not used
    else:
        projector = Projector(args.calib)
    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)
    
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    if args.video_save_filedir is not None:
        import ffmpeg_util, timestamp_util
        video_filepath = os.path.join(args.video_save_filedir, timestamp_util.global_timestamp_str + ".mkv")
        video_save = ffmpeg_util.VideoOut(out_path=video_filepath, video_shape=(1280, 720), fps=5)
    else:
        video_save = None
    try:
        with torch.inference_mode():
            policy.eval()
            prev_width = None
            for t in range(args.max_steps):
                if t % args.num_inference_step == 0:
                    qpos_tcp = agent.get_tcp_pose()[None]
                    qpos_tcp = xyz_rot_transform(qpos_tcp, from_rep = "quaternion", to_rep = "rotation_6d")
                    qpos_tcp = projector.project_tcp_to_camera_coord(qpos_tcp, cam = agent.camera_serial, rotation_rep = "rotation_6d")
                    qpos_gripper_width = decode_gripper_width(agent.get_gripper_width())[None]
                    qpos = np.concatenate((qpos_tcp, qpos_gripper_width), axis=-1)
                    qpos = qpos.astype(np.float32)
                    qpos = normalize_tcp(qpos)
                    qpos = torch.from_numpy(qpos).to(device)
                    # pre-process inputs
                    colors, depths = agent.get_observation()
                    if args.vis:
                        coords, feats, cloud = create_input(
                            colors,
                            depths,
                            cam_intrinsics = agent.intrinsics,
                            voxel_size = args.voxel_size
                        )
                        ## feats, coords = feats.to(device), coords.to(device)
                        ## cloud_data = ME.SparseTensor(feats, coords)
                    colors_hand, depths_hand = agent.get_observation_hand()
                    if args.center_crop:
                        colors = torch.from_numpy(colors).unsqueeze(0).permute(0, 3, 1, 2)
                        colors_hand = torch.from_numpy(colors_hand).unsqueeze(0).permute(0, 3, 1, 2)
                        colors = center_crop(colors)
                        colors_hand = center_crop(colors_hand)
                        colors = colors.permute(0, 2, 3, 1).squeeze(0).numpy()
                        colors_hand = colors_hand.permute(0, 2, 3, 1).squeeze(0).numpy()
                        # print(colors.shape)
                    if args.vis:
                        import cv2
                        cv2.namedWindow("topdown")
                        cv2.namedWindow("inhand")
                        while True:
                            cv2.imshow("topdown", colors[..., ::-1])
                            cv2.imshow("inhand", colors_hand[..., ::-1])
                            key = cv2.waitKey(1)
                            if key == ord('q'):
                                break
                        cv2.destroyWindow("topdown")
                        cv2.destroyWindow("inhand")
                    colors = torch.from_numpy(colors).float().to(device)
                    colors = colors.unsqueeze(0).unsqueeze(1)
                    colors_hand = torch.from_numpy(colors_hand).float().to(device)
                    colors_hand = colors_hand.unsqueeze(0).unsqueeze(1)
                    # predict
                    pred_raw_action = policy(
                        imgtop=colors,
                        imghand=colors_hand,
                        actions = None,
                        qpos = qpos,
                        batch_size = 1,
                    ).squeeze(0).cpu().numpy()
                    # unnormalize predicted actions
                    if not args.no_project:
                        action = unnormalize_action(pred_raw_action)
                    else:
                        action = unnormalize_action_noproject(pred_raw_action)
                    # visualization
                    if args.vis:
                        import open3d as o3d
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                        pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                        # project action to camera coordinate
                        if not args.no_project:
                            action_tcp_proj = action[..., :-1]
                        else:
                            action_tcp_proj = projector.project_tcp_to_camera_coord(action[..., :-1], cam = agent.camera_serial, rotation_rep = "rotation_6d")
                        tcp_vis_list = []
                        for raw_tcp in action_tcp_proj:
                            tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                            tcp_vis_list.append(tcp_vis)
                        o3d.visualization.draw_geometries([pcd, *tcp_vis_list])
                    if not args.no_project:
                        action_tcp = projector.project_tcp_to_base_coord(action[..., :-1], cam = agent.camera_serial, rotation_rep = "rotation_6d")
                    else:
                        action_tcp = action[..., :-1]
                    action_width = action[..., -1]
                    # safety insurance
                    action_tcp[..., :3] = np.clip(action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
                    # full actions
                    action = np.concatenate([action_tcp, action_width[..., np.newaxis]], axis = -1)
                    # add to ensemble buffer
                    ensemble_buffer.add_action(action, t)

                # save video at each action
                if video_save is not None:
                    video_save.write(agent.camera.get_rgbd_image()[0])
                
                # get step action from ensemble buffer
                step_action = ensemble_buffer.get_action()
                if step_action is None:   # no action in the buffer => no movement.
                    continue
                
                step_tcp = step_action[:-1]
                step_width = step_action[-1]

                # send tcp pose to robot
                if args.discretize_rotation:
                    rot_steps = discretize_rotation(last_rot, step_tcp[3:], np.pi / 16)
                    last_rot = step_tcp[3:]
                    for rot in rot_steps:
                        step_tcp[3:] = rot
                        agent.set_tcp_pose(
                            step_tcp, 
                            rotation_rep = "rotation_6d",
                            blocking = True
                        )
                else:
                    agent.set_tcp_pose(
                        step_tcp,
                        rotation_rep = "rotation_6d",
                        blocking = True
                    )
                
                # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
                if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                    agent.set_gripper_width(step_width, blocking = True)
                    prev_width = step_width
    except KeyboardInterrupt:
        pass
    
    agent.stop()
    if video_save is not None:
        video_save.close()

def decode_gripper_width(gripper_width):
    # return gripper_width / 1000. * 0.095
    # robotiq-85: 0.0000 - 0.0085
    #                255 -      0
    return (1. - gripper_width / 255.) * 0.085

def normalize_tcp(tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = True)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 20)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')

    parser.add_argument('--enable_mba', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--obj_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 9)
    parser.add_argument('--video_save_filedir', action = 'store', type = str, help = 'video save path', required = False, default=None)

    parser.add_argument('--no_project', action= 'store_true', help = 'turn off projector')
    parser.add_argument('--projector_mocap_mode', action = 'store_true', help = 'use mocap mode of projector (percomputed calib info)')

    parser.add_argument('--center_crop', action = 'store_true', help='use center crop')

    evaluate(vars(parser.parse_args()))