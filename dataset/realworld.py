import os
import json
import torch
import numpy as np
import open3d as o3d
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from termcolor import cprint

from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

NUM_DEMO = 47
RECORD_METAINFO_FILEDIR = "data/record_metainfo"
CALIB_FILEDIR = "data/common/calib_mocap/main/rise_calib"
MOCAP_FILEDIR = "data/mocap_data"

import pickle
from .sync import millisec_to_timestamp, sync_stream, load_mocap_csv_timestamp, load_mocap_csv, timestamp_to_millisec

class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['104122060902'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        with_obj_action = False,
        hand_cam_id = 104422070044,
        forval = False,
        no_project = False,
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        # self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.with_obj_action = with_obj_action
        self.hand_cam_id = hand_cam_id
        
        if not forval: 
            self.all_demos = sorted(os.listdir(self.data_path))[:NUM_DEMO] 
            assert len(self.all_demos) == NUM_DEMO, f"{len(self.all_demos)}"
        else:
            self.all_demos = sorted(os.listdir(self.data_path))[NUM_DEMO:NUM_DEMO+1]
        self.num_demos = len(self.all_demos)
        
        # custom fields
        self.task_prefix = os.path.basename(self.path)
        with open(os.path.join(RECORD_METAINFO_FILEDIR, "teleop_to_calib.json"), "rb") as ifs:
            self.teleop_to_calib = json.load(ifs)
        _camonly_calib = os.path.join(RECORD_METAINFO_FILEDIR, "teleop_to_camcalib.json")
        if os.path.exists(_camonly_calib):
            with open(_camonly_calib, "rb") as ifs:
                self.teleop_to_camcalib = json.load(ifs)
        else:
            self.teleop_to_camcalib = None
        if self.with_obj_action:
            with open(os.path.join(RECORD_METAINFO_FILEDIR, "teleop_to_mocap.json"), "rb") as ifs:
                self.teleop_to_mocap = json.load(ifs)
            with open(os.path.join(RECORD_METAINFO_FILEDIR, "teleop_to_obj.json"), "rb") as ifs:
                self.teleop_to_obj = json.load(ifs)
            with open(os.path.join(RECORD_METAINFO_FILEDIR, "obj_reg.pkl"), "rb") as ifs:
                self.obj_reg = pickle.load(ifs)

        # no project
        self.no_project = no_project

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.hand_frame_ids = []
        self.action_frame_ids = []
        if self.with_obj_action:
            self.pk_list = []
            self.obj_frame_ids = []
            self.mocap_csv = {}
            self.mocap_ts = {}
            self.T_robot_mocap = {}
        self.projectors = {}
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata
                try:
                    with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                        meta = json.load(f)
                except FileNotFoundError:
                    meta = None
                # get frame ids
                if meta is not None:
                    finish_time = meta["finish_time"]
                else: # use heuristic, omit last 1 second of data # TODO: manual finish_time
                    finish_time = sorted(os.listdir(os.path.join(cam_path, "color")), reverse=True)[0]
                    finish_time = int(os.path.splitext(finish_time)[0]) - 1000
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= finish_time
                ]
                hand_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join( os.path.join(demo_path, "cam_{}".format(hand_cam_id)), "color"))) 
                    if int(os.path.splitext(x)[0]) <= finish_time
                ]

                pk = f"{self.task_prefix}/{self.all_demos[i]}"
                if self.path != "data/rise/pour_2": 
                    calib_filepath = self.teleop_to_calib[pk]
                else:
                    calib_filepath = self.teleop_to_camcalib[pk]

                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                if self.with_obj_action:
                    obj_frame_ids_list = []
                hand_frame_ids_list= []
                hand_ts =  [millisec_to_timestamp(ms) for ms in hand_ids]
                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1) #0
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx)) # action after end
                    frame_begin = max(0, cur_idx - num_obs + 1) #cur id
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                   
                    obs_ts = [millisec_to_timestamp(ms) for ms in obs_frame_ids]
                    obs_stream_sync =  sync_stream(
                            {"obs": obs_ts, "hand": hand_ts},
                            ref_stream_timestamp=obs_ts
                        )
                    hand_frame_ids = obs_stream_sync["hand"]
                    hand_frame_ids = [timestamp_to_millisec(ms) for ms in hand_frame_ids]
                    hand_frame_ids_list.append(hand_frame_ids) 
                    if self.with_obj_action:
                        # load mocap timestamp
                        stream_ts = [millisec_to_timestamp(ms) for ms in action_frame_ids]
                        if pk not in self.mocap_ts:
                            mocap_ts = load_mocap_csv_timestamp(os.path.join(MOCAP_FILEDIR, self.teleop_to_mocap[pk]))
                            self.mocap_ts[pk] = mocap_ts
                        else:
                            mocap_ts = self.mocap_ts[pk]

                        # sync stream to get obj_frame_ids_list
                        multi_stream_sync =  sync_stream(
                            {"action": stream_ts, "mocap": mocap_ts},
                            ref_stream_timestamp=stream_ts
                        )
                        obj_frame_ids = multi_stream_sync["mocap"]
                        
                        obj_frame_ids_list.append(obj_frame_ids)
                        if pk not in self.T_robot_mocap:
                            # load current T_robot_mocap
                            with open(os.path.join(CALIB_FILEDIR, self.teleop_to_calib[pk], "calib_info.pkl"), "rb") as ifs:
                                calib_info = pickle.load(ifs)
                            self.T_robot_mocap[pk] = calib_info["T_robot_mocap"]

                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_filepath] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.hand_frame_ids += hand_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
                if self.with_obj_action:
                    self.pk_list += [pk] * len(obj_frame_ids_list)
                    self.obj_frame_ids += obj_frame_ids_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list
    
    def _normalize_obj(self, obj_list):
        ''' obj_list: [T, 3(trans) + 6(rot)]'''
        obj_list[:, :3] = (obj_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        return obj_list
    
    def _normalize_tcp_noproject(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - SAFE_WORKSPACE_MIN) / (SAFE_WORKSPACE_MAX - SAFE_WORKSPACE_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list
    
    def _normalize_obj_noproject(self, obj_list):
        ''' obj_list: [T, 3(trans) + 6(rot)]'''
        obj_list[:, :3] = (obj_list[:, :3] - SAFE_WORKSPACE_MIN) / (SAFE_WORKSPACE_MAX - SAFE_WORKSPACE_MIN) * 2 - 1
        return obj_list

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        hand_frame_ids = self.hand_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        hand_color_dir = os.path.join(data_path, "cam_{}".format(self.hand_cam_id), 'color')

        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')

       
        timestamp = calib_timestamp
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(CALIB_FILEDIR, timestamp), mocap_mode=True)
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.Compose([
                T.RandomPerspective(distortion_scale=0.2),
                T.RandomResizedCrop((200,256), scale=(0.6, 1.0), ratio=(1.0, 1.0)),
                T.RandomApply([
                T.ColorJitter(
                brightness=self.aug_jitter_params[0],
                contrast=self.aug_jitter_params[1],
                saturation=self.aug_jitter_params[2],
                hue=self.aug_jitter_params[3]
                )
                ], p=self.aug_jitter_prob)
                ])

        # load colors and depths
        colors_list = []
        hand_colors_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
        colors_list = np.stack(colors_list, axis = 0)
        
        for frame_id in hand_frame_ids:
            colors = Image.open(os.path.join(hand_color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            hand_colors_list.append(colors)
        hand_colors_list = np.stack(hand_colors_list, axis = 0)



        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            if not self.no_project:
                projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            else:
                projected_tcp = tcp
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        if self.with_obj_action:
            pk = self.pk_list[index]
            # mocap_csv
            if pk not in self.mocap_csv:
                mocap_tsposevec = load_mocap_csv(os.path.join(MOCAP_FILEDIR, self.teleop_to_mocap[pk]), timestamp_list=self.mocap_ts[pk])
                self.mocap_csv[pk] = mocap_tsposevec
            mocap_tsposevec = self.mocap_csv[pk]
            # obj & reg_transf
            obj = self.teleop_to_obj[pk]["obj_mocap"]
            obj_reg = self.obj_reg[obj]
            T_robot_mocap = self.T_robot_mocap[pk]
            # obj_actions
            obj_frame_ids = self.obj_frame_ids[index]
            action_objs = []
            for frame_id in obj_frame_ids:
                posevec_map = mocap_tsposevec[frame_id]
                obj_posevec = posevec_map[obj]
                obj_transf_mocap = xyz_rot_transform(obj_posevec, from_rep="quaternion", to_rep="matrix")
                obj_transf_robot = T_robot_mocap @ obj_transf_mocap @ obj_reg
                obj_posevec = xyz_rot_transform(obj_transf_robot, from_rep="matrix", to_rep="quaternion")
                if not self.no_project:
                    projected_obj = projector.project_tcp_to_camera_coord(obj_posevec, cam_id)
                else:
                    projected_obj = obj_posevec
                action_objs.append(projected_obj)
            action_objs = np.stack(action_objs)

        
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)

        # normalization
        if not self.no_project:
            actions_normalized = self._normalize_tcp(actions.copy())
        else:
            actions_normalized = self._normalize_tcp_noproject(actions.copy())
            # actions_normalized = actions.copy()

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'colors_list': colors_list,
            'hand_colors_list': hand_colors_list,
            'action': actions,
            'action_normalized': actions_normalized
        }

        if self.with_obj_action:
            action_objs = xyz_rot_transform(action_objs, from_rep="quaternion", to_rep="rotation_6d")
            actions_obj = action_objs
            if not self.no_project:
                actions_obj_normalized = self._normalize_obj(actions_obj.copy())
            else:
                actions_obj_normalized = self._normalize_obj_noproject(actions_obj.copy())
            actions_obj_normalized = torch.from_numpy(actions_obj_normalized).float()
            ret_dict["action_obj_normalized"] = actions_obj_normalized

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    # return gripper_width / 1000. * 0.095
    # robotiq-85: 0.0000 - 0.0085
    #                255 -      0
    return (1. - gripper_width / 255.) * 0.085

