import h5py
import numpy as np
import torch
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from depth_anything_3.api import DepthAnything3
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import zoom
import shutil
import cv2
import roboticstoolbox as rtb
from scipy.spatial import KDTree
import pdb


class CondGenerator:
    def __init__(
        self,
        model_name: str = "../3d_control/depth-anything/DA3NESTED-GIANT-LARGE",
        model_path: str = "/mnt/data-2/users/wangboyuan/xxw/3d_control/Depth-Anything-3/checkpoints",
        urdf_path: str = "/mnt/data-2/users/wangboyuan/xxw/3d_control/robot_utils/piper_twin.urdf",
        gripper_mesh_dir: str = "/mnt/data-2/users/wangboyuan/xxw/3d_control/robot_utils/piper/meshes",
        device: str = "cuda"
    ):
        """
        初始化条件生成器
        
        Args:
            model_name: DA3模型名称
            model_path: DA3模型路径
            urdf_path: 机器人URDF文件路径
            gripper_mesh_dir: gripper STL文件目录
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🚀 初始化条件生成器")
        print(f"  设备: {self.device}")
        
        # 加载DA3模型
        self.model = DepthAnything3.from_pretrained(model_path)
        self.model = self.model.to(device=self.device)
        print(f"✓ DA3模型加载完成")
        
        # 加载机器人运动学
        self.urdf_path = urdf_path
        self.gripper_mesh_dir = gripper_mesh_dir
        self.setup_robot_kinematics()
        
        # 设置相机参数
        self.setup_camera_params()
        
        # 碰撞检测阈值 (单位: 米)
        self.collision_threshold = 0.01
        self.scene_density_threshold = 20
        self.gripper_contact_min_points = 1

        # Wrist camera gripper GT 数据库（用于 gripper-proj 与 gripper-gt 匹配）
        # TODO: 指定数据库路径并加载
        self.gripper_gt_db = None

        print(f"✓ 初始化完成\n")

    def setup_robot_kinematics(self):
        """设置机器人运动学链"""
        print(f"🤖 加载机器人URDF: {self.urdf_path}")
        
        # 左臂和右臂的末端link名
        END_LINK_LEFT = "camera"
        END_LINK_RIGHT = "camera"
        END_LINK_LEFT_GRIPPER1 = "link7"
        END_LINK_LEFT_GRIPPER2 = "link8"
        END_LINK_RIGHT_GRIPPER1 = "link7"
        END_LINK_RIGHT_GRIPPER2 = "link8"
        
        # 加载机器人
        links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(self.urdf_path)
        robot = rtb.Robot(links, name=name, manufacturer="Piper",
                        urdf_string=urdf_string, urdf_filepath=urdf_filepath)
        
        # 创建运动学链
        self.ets_left = robot.ets(end=END_LINK_LEFT)
        self.ets_right = robot.ets(end=END_LINK_RIGHT)
        self.ets_left_gripper1 = robot.ets(end=END_LINK_LEFT_GRIPPER1)
        self.ets_left_gripper2 = robot.ets(end=END_LINK_LEFT_GRIPPER2)
        self.ets_right_gripper1 = robot.ets(end=END_LINK_RIGHT_GRIPPER1)
        self.ets_right_gripper2 = robot.ets(end=END_LINK_RIGHT_GRIPPER2)
        
        # Eye-in-hand 外参
        self.T_cam2gripper_left = np.array([
            [-0.02500583,  0.93433644,  0.35551388,  0.0242712],
            [-0.99959639, -0.02816514,  0.00371289,  0.01099694],
            [ 0.01348219, -0.35527755,  0.93466363,  0.18571905],
            [0, 0, 0, 1]
        ])
        
        self.T_cam2gripper_right = np.array([
            [8.33993823e-04,  9.32149133e-01,  3.62073608e-01,  0.00512849],
            [-9.99178236e-01,  1.54492857e-02, -3.74722971e-02, -0.00114925],
            [-4.05235479e-02, -3.61744817e-01,  9.31396011e-01,  0.15770671],
            [0, 0, 0, 1]
        ])
        
        # 坐标系修正矩阵
        self.T_coord_left = np.array([
            [-0.92679400, -0.36929676, 0.06835755, -0.01527710],
            [-0.06590783, -0.01926177, -0.99763981, 0.01499681],
            [0.36974188, -0.92911181, -0.00648783, 0.25386413],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        ]) @ np.array([
            [9.99390255e-01,  1.40550761e-07, -3.49010839e-02, 5.71904409e-03],
            [1.47184143e-07,  1.00000280e+00, -2.75173177e-08, 3.40000464e-02],
            [3.48978522e-02, -1.64221341e-08,  9.99389886e-01, -1.72866273e-03],
            [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
        ])
        
        self.T_coord_right = np.array([
            [-0.92672129, -0.37572924, -0.00390775, -0.06222343],
            [0.01810535, -0.03426396, -0.99924890, -0.00574049],
            [0.37531297, -0.92609582, 0.03855580, 0.18249571],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        ]) @ np.array([
            [0.99870762,  0.00120439,  0.0677579 , -0.02097595],
            [ 0.01627267,  0.99103597, -0.11658521,  0.04128389],
            [-0.05798717,  0.1249512 ,  0.99532125,  0.04534942],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])

        # gripper坐标系修正矩阵
        # self.T_gripper_left1 = np.array([
        #     [ 9.95001405e-01,  2.52755838e-02,  9.66092988e-02, 2.50780280e-02],
        #     [-2.54623019e-02,  9.99675540e-01,  7.00145320e-04, 9.21394677e-04],
        #     [-9.65602546e-02, -3.15653234e-03,  9.95322135e-01, -1.69249530e-02],
        #     [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
        # ])
        # self.T_gripper_left2 = np.array([
        #     [ 0.99516232, -0.02477129, -0.09507019,  0.02485217],
        #     [ 0.02648465,  0.999508  ,  0.01680252, -0.00539399],
        #     [ 0.0946072 , -0.01923913,  0.99532875,  0.01585377],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]
        # ])

        # self.T_gripper_right1 = np.array([
        #     [ 0.99593533,  0.03462796,  0.08314877, -0.01924522],
        #     [-0.0309955 ,  0.99852456, -0.04458701,  0.01223784],
        #     [-0.08457004,  0.04182855,  0.99553919, -0.00838989],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]
        # ])
        # self.T_gripper_right2 = np.array([
        #     [ 0.99541578, -0.03317571, -0.08970398, -0.01928617],
        #     [ 0.03057097,  0.99907454, -0.03025713, -0.00768509],
        #     [ 0.09062476,  0.02737609,  0.99550876,  0.01044267],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]
        # ])
        self.T_gripper_left1 = np.load("../3d_control/trans_mat/link7_T_corr_2.npy")
        self.T_gripper_left2 = np.load("../3d_control/trans_mat/link8_T_corr_2.npy")
        self.T_gripper_right1 = np.load("../3d_control/trans_mat/link7_T_corr_2.npy")
        self.T_gripper_right2 = np.load("../3d_control/trans_mat/link8_T_corr_2.npy")
        
        # Front相机 -> 左/右臂base
        self.T_front2basel = np.array([
            [ 0.05831506, -0.84520743,  0.53124736,  0.02381213],
            [-0.99752094, -0.02833829,  0.06441209, -0.34711892],
            [-0.03938694, -0.53368656, -0.84476466,  0.66712113],
            [0, 0, 0, 1]
        ]) 
        
        self.T_front2baser = np.array([
            [-0.00946253, -0.84779082,  0.53024634,  0.01105757],
            [-0.99886042,  0.03282072,  0.03465061,  0.25614093],
            [-0.04677953, -0.52931420, -0.84713527,  0.63708367],
            [0, 0, 0, 1]
        ])

        # 三视角单独投影校正矩阵
        self.T_gripper2cam_mat = {}
        self.T_gripper2cam_mat["left"] = np.load("../3d_control/trans_mat/left_trans_mat.npy")
        self.T_gripper2cam_mat["right"] = np.load("../3d_control/trans_mat/right_trans_mat.npy")
        self.T_gripper2cam_mat["front"] = np.load("../3d_control/trans_mat/front_trans_mat.npy")
        
        print(f"✓ 运动学链加载完成")

    def setup_camera_params(self):
        """设置相机参数"""
        # RGB相机内参
        self.intrinsics = {
            'left': np.array([
                [605.4948120117188, 0.0, 325.0260925292969],
                [0.0, 605.5114135742188, 246.6322479248047],
                [0.0, 0.0, 1.0]
            ]),
            'right': np.array([
                [607.4896850585938, 0.0, 332.4833984375],
                [0.0, 606.8885498046875, 249.5357666015625],
                [0.0, 0.0, 1.0]
            ]),
            'front': np.array([
                [488.615234375, 0.0, 321.0052185058594],
                [0.0, 488.615234375, 217.4329071044922],
                [0.0, 0.0, 1.0]
            ])
        }
        
        self.camera_names = ['front', 'left', 'right']

    def _fk_base_to_end(self, ets, q_arm: np.ndarray) -> np.ndarray:
        """
        FK计算: base -> 末端
        
        Args:
            ets: 运动学链
            q_arm: 关节角度 [N, n_joints]
        
        Returns:
            T_list: 变换矩阵列表 [N, 4, 4]
        """
        N = q_arm.shape[0]
        n_joints = ets.n
        
        if q_arm.shape[1] != n_joints:
            raise ValueError(f"关节角度维度 {q_arm.shape[1]} 与链的自由度 {n_joints} 不匹配")
        
        T_list = np.empty((N, 4, 4), dtype=np.float64)
        for i in range(N):
            T = ets.fkine(q_arm[i])
            T_list[i] = T.A
        return T_list

    def _load_gripper_mesh(self, link_name: str) -> np.ndarray:
        """
        加载gripper的STL网格并转换为点云
        
        Args:
            link_name: "link7" 或 "link8"
        
        Returns:
            points: [N, 3] numpy数组，网格顶点坐标
        """
        import trimesh
        mesh_path = os.path.join(self.gripper_mesh_dir, f"{link_name}.STL")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Gripper mesh文件不存在: {mesh_path}")
        
        # 加载STL网格
        mesh = trimesh.load(mesh_path)
        
        # 在表面均匀采样更多点（推荐，点云更密集）
        points, _ = trimesh.sample.sample_surface(mesh, count=5000)
        
        return points

    def forward_kinematics(self, current_action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        通过FK计算相机外参
        
        Args:
            current_action: 当前时刻的qpos [14,] (左臂7 + 右臂7)
        
        Returns:
            extrinsics: 字典，包含三个相机的W2C外参
        """
        qpos = current_action.reshape(1, -1)  # [1, 14]
        
        q_left7 = qpos[:, :7]
        q_right7 = qpos[:, -7:]
        q_left6 = q_left7[:, :6].astype(np.float64)
        q_right6 = q_right7[:, :6].astype(np.float64)
        
        # FK: base -> gripper
        T_basel2gripper = self._fk_base_to_end(self.ets_left, q_left6)[0]
        T_baser2gripper = self._fk_base_to_end(self.ets_right, q_right6)[0]
        
        # base -> front
        T_F_basel = np.linalg.inv(self.T_front2basel)
        T_F_baser = np.linalg.inv(self.T_front2baser)
        
        # front -> gripper
        T_F_gripper_left = T_F_basel @ T_basel2gripper
        T_F_gripper_right = T_F_baser @ T_baser2gripper
        
        # gripper -> camera (手眼变换)
        T_gripper2cam_left = np.linalg.inv(self.T_cam2gripper_left)
        T_gripper2cam_right = np.linalg.inv(self.T_cam2gripper_right)
        
        # front -> camera
        T_F_cam_left = T_F_gripper_left @ T_gripper2cam_left
        T_F_cam_right = T_F_gripper_right @ T_gripper2cam_right
        
        # 坐标系修正
        T_F_cam_left = T_F_cam_left @ self.T_coord_left
        T_F_cam_right = T_F_cam_right @ self.T_coord_right
        
        # W2C
        extrinsics = {
            'front': np.eye(4),
            'left': np.linalg.inv(T_F_cam_left),
            'right': np.linalg.inv(T_F_cam_right)
        }
        
        return extrinsics

    def forward_DA3(
        self, 
        current_obs: List[np.ndarray], 
        extrinsics: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        运行DA3推理获取深度图
        
        Args:
            current_obs: 三视角RGB图像列表 [front_rgb, left_rgb, right_rgb]
            extrinsics: 三视角外参字典
        
        Returns:
            depths: 深度图字典
            extrinsics: 外参字典
            intrinsics: 内参字典
        """
        # 准备输入
        images_array = []
        intrinsics_list = []
        extrinsics_list = []
        
        for cam in self.camera_names:
            idx = self.camera_names.index(cam)
            images_array.append(current_obs[idx])
            intrinsics_list.append(self.intrinsics[cam])
            extrinsics_list.append(extrinsics[cam])
        
        intrinsics_array = np.stack(intrinsics_list, axis=0)
        extrinsics_array = np.stack(extrinsics_list, axis=0)
        
        # DA3推理
        prediction = self.model.inference(
            image=images_array,
            intrinsics=intrinsics_array,
            extrinsics=extrinsics_array,
            use_ray_pose=True,
            infer_gs=False
        )
        
        # 组织结果
        depths = {}
        extrinsics_out = {}
        intrinsics_out = {}
        
        for i, cam in enumerate(self.camera_names):
            depths[cam] = prediction.depth[i]
            extrinsics_out[cam] = prediction.extrinsics[i]
            intrinsics_out[cam] = prediction.intrinsics[i]
        
        return depths, extrinsics_out, intrinsics_out

    def convert_depth(
        self,
        current_obs: List[np.ndarray],
        depths: Dict[str, np.ndarray],
        extrinsics: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        将深度图转换为点云（世界坐标系 - front camera）
        
        Returns:
            front_pts, left_pts, right_pts: 三个相机的点云
            arm_masks: Dict[str, np.ndarray]，每个相机的机械臂mask
        """
        points_dict = {}
        arm_masks = {}
        
        for i, cam in enumerate(self.camera_names):
            rgb = current_obs[i]
            depth = depths[cam]
            intrinsic = intrinsics[cam]
            extrinsic = extrinsics[cam]
            
            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
            
            mask_arm = self._detect_arm_pixels(rgb)
            arm_masks[cam] = mask_arm
            
            print(f"  {cam}: 机械臂像素 {mask_arm.sum()} / {mask_arm.size} "
                  f"({mask_arm.sum()/mask_arm.size*100:.1f}%)")
            
            points = self._depth_to_pointcloud(
                depth=depth,
                intrinsic=intrinsic,
                rgb=rgb,
                mask_exclude=mask_arm
            )
            
            if cam != 'front':
                points = self._transform_to_world(points, extrinsic)
            
            points_dict[cam] = points
        
        return points_dict['front'], points_dict['left'], points_dict['right'], arm_masks

    def convert_depth_with_flow_mask(
        self,
        current_obs: List[np.ndarray],
        depths: Dict[str, np.ndarray],
        extrinsics: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray],
        flow_lg1_2D: Dict[str, np.ndarray],
        flow_lg2_2D: Dict[str, np.ndarray],
        flow_rg1_2D: Dict[str, np.ndarray],
        flow_rg2_2D: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],
               List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        深度图转点云 + 基于 flow mask 重建 K+1 帧 gripper 点云（2D 对齐专用）

        与 convert_depth 的区别:
        - front camera: 仍用颜色过滤去除机械臂像素（gripper 被机械臂遮挡不可见）
        - left/right wrist camera: 使用上游 flow mask 区分场景 vs gripper 像素
          - 场景像素 → 转为场景点云
          - gripper 像素 → 利用 wrist depth 反投影为 T=0 gripper 点云，
            再结合 3D flow 得到 K+1 帧 gripper 点云

        输出格式与 3D 方案的 get_gripper_points 完全一致:
        每个 gripper 返回 List[np.ndarray]，长度 K+1，每个 [N, 3]，
        均在 front camera 坐标系（世界坐标系）下。

        Args:
            current_obs: 三视角 RGB [front, left, right]
            depths: 深度图字典 {'front', 'left', 'right'}
            extrinsics: 外参字典
            intrinsics: 内参字典
            flow_lg1_2D: 左臂 gripper1 的 2D flow, dict with 'mask' [H,W] bool, 'flow' [H,W,K,3]
            flow_lg2_2D: 左臂 gripper2 的 2D flow
            flow_rg1_2D: 右臂 gripper1 的 2D flow
            flow_rg2_2D: 右臂 gripper2 的 2D flow

        Returns:
            front_pts: front camera 场景点云 [N, 6] (xyz+rgb)
            left_pts: left camera 场景点云（去除 gripper 区域）[N, 6]
            right_pts: right camera 场景点云（去除 gripper 区域）[N, 6]
            arm_masks: 机械臂 mask 字典
            gripper_pts_lg1: List[np.ndarray], 左臂 gripper1 点云列表, 长度 K+1, 每个 [N, 3], front camera 坐标系
            gripper_pts_lg2: List[np.ndarray], 左臂 gripper2 点云列表, 长度 K+1, 每个 [N, 3], front camera 坐标系
            gripper_pts_rg1: List[np.ndarray], 右臂 gripper1 点云列表, 长度 K+1, 每个 [N, 3], front camera 坐标系
            gripper_pts_rg2: List[np.ndarray], 右臂 gripper2 点云列表, 长度 K+1, 每个 [N, 3], front camera 坐标系

        """
        K = flow_lg1_2D['flow'].shape[2]

        points_dict = {}
        arm_masks = {}

        # --- Front camera: 颜色过滤机械臂（与 convert_depth 相同）---
        rgb_front = current_obs[0]
        depth_front = depths['front']
        intr_front = intrinsics['front']
        ext_front = extrinsics['front']
        if ext_front.shape == (3, 4):
            ext_front = np.vstack([ext_front, [0, 0, 0, 1]])

        mask_arm_front = self._detect_arm_pixels(rgb_front)
        arm_masks['front'] = mask_arm_front

        front_pts = self._depth_to_pointcloud(
            depth_front, intr_front, rgb_front, mask_exclude=mask_arm_front
        )
        # front = world，无需变换
        points_dict['front'] = front_pts

        print(f"  front: 场景点 {len(front_pts)}, "
              f"机械臂像素 {mask_arm_front.sum()} / {mask_arm_front.size} "
              f"({mask_arm_front.sum()/mask_arm_front.size*100:.1f}%)")

        # --- Left wrist camera: flow mask 区分场景/gripper ---
        rgb_left = current_obs[1]
        depth_left = depths['left']
        intr_left = intrinsics['left']
        ext_left = extrinsics['left']
        if ext_left.shape == (3, 4):
            ext_left = np.vstack([ext_left, [0, 0, 0, 1]])

        gripper_mask_left = flow_lg1_2D['mask'] | flow_lg2_2D['mask']
        arm_masks['left'] = gripper_mask_left

        # 场景点云：排除 gripper 区域（resize mask 到 depth 分辨率）
        H_depth_l, W_depth_l = depth_left.shape
        if gripper_mask_left.shape[:2] != (H_depth_l, W_depth_l):
            gripper_mask_left_rs = cv2.resize(
                gripper_mask_left.astype(np.uint8), (W_depth_l, H_depth_l),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gripper_mask_left_rs = gripper_mask_left

        left_scene_pts = self._depth_to_pointcloud(
            depth_left, intr_left, rgb_left, mask_exclude=gripper_mask_left_rs
        )
        left_scene_pts = self._transform_to_world(left_scene_pts, ext_left)
        points_dict['left'] = left_scene_pts

        # Gripper 点云：depth 反投影 + flow 重建 K+1 帧
        C2W_left = np.linalg.inv(ext_left)
        gripper_pts_lg1 = self._reconstruct_gripper_from_flow(
            depth_left, intr_left, C2W_left, flow_lg1_2D, K
        )
        gripper_pts_lg2 = self._reconstruct_gripper_from_flow(
            depth_left, intr_left, C2W_left, flow_lg2_2D, K
        )

        print(f"  left: 场景点 {len(left_scene_pts)}, "
              f"gripper像素 {gripper_mask_left.sum()}, "
              f"lg1点数 {len(gripper_pts_lg1[0])}, lg2点数 {len(gripper_pts_lg2[0])}")

        # --- Right wrist camera: 同 left ---
        rgb_right = current_obs[2]
        depth_right = depths['right']
        intr_right = intrinsics['right']
        ext_right = extrinsics['right']
        if ext_right.shape == (3, 4):
            ext_right = np.vstack([ext_right, [0, 0, 0, 1]])

        gripper_mask_right = flow_rg1_2D['mask'] | flow_rg2_2D['mask']
        arm_masks['right'] = gripper_mask_right

        H_depth_r, W_depth_r = depth_right.shape
        if gripper_mask_right.shape[:2] != (H_depth_r, W_depth_r):
            gripper_mask_right_rs = cv2.resize(
                gripper_mask_right.astype(np.uint8), (W_depth_r, H_depth_r),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gripper_mask_right_rs = gripper_mask_right

        right_scene_pts = self._depth_to_pointcloud(
            depth_right, intr_right, rgb_right, mask_exclude=gripper_mask_right_rs
        )
        right_scene_pts = self._transform_to_world(right_scene_pts, ext_right)
        points_dict['right'] = right_scene_pts

        C2W_right = np.linalg.inv(ext_right)
        gripper_pts_rg1 = self._reconstruct_gripper_from_flow(
            depth_right, intr_right, C2W_right, flow_rg1_2D, K
        )
        gripper_pts_rg2 = self._reconstruct_gripper_from_flow(
            depth_right, intr_right, C2W_right, flow_rg2_2D, K
        )

        print(f"  right: 场景点 {len(right_scene_pts)}, "
              f"gripper像素 {gripper_mask_right.sum()}, "
              f"rg1点数 {len(gripper_pts_rg1[0])}, rg2点数 {len(gripper_pts_rg2[0])}")

        return (points_dict['front'], points_dict['left'], points_dict['right'],
                arm_masks,
                gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)

    def _detect_arm_pixels(self, rgb: np.ndarray, threshold: int = 30) -> np.ndarray:
        """
        检测机械臂像素（HSV空间：低亮度 + 低饱和度的黑色区域）
        """
        # 转换为灰度图
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 转换为HSV图
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # 黑色机械臂: 低V（亮度）且低S（饱和度）
        # 这排除了深色但有颜色的物体（如深蓝桌面）
        mask_dark = hsv[:, :, 2] < 50       # 亮度低
        mask_low_sat = hsv[:, :, 1] < 80    # 饱和度低（纯黑色饱和度接近0）
        
        mask_arm = (mask_dark & mask_low_sat) | (gray < threshold)
        
        # 形态学操作：去噪 + 填充 + 膨胀
        kernel = np.ones((5, 5), np.uint8)
        mask_arm = cv2.morphologyEx(mask_arm.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_arm = cv2.morphologyEx(mask_arm, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask_arm = cv2.dilate(mask_arm, kernel, iterations=1)
        
        return mask_arm.astype(bool)

    def _lookup_gripper_gt_mask(
        self,
        action: np.ndarray,
        cam_name: str,
        image_size: Tuple[int, int] = (480, 640)
    ) -> np.ndarray:
        """
        从预构建的 wrist camera 数据库中检索该 action 对应的 gripper 像素 mask (gripper-gt)

        数据库存储了不同 action 下各相机中可见的 gripper 像素区域。
        通过最近邻检索找到最匹配的 action，返回对应的 gripper binary mask。

        Args:
            action: [14,] 当前帧 qpos
            cam_name: 相机名称 ('left', 'right', 'front')
            image_size: 图像尺寸 (H, W)

        Returns:
            gripper_gt_mask: [H, W] bool, GT gripper 像素 mask

        TODO:
            - 定义数据库格式和存储路径（self.gripper_gt_db）
            - 实现 action 空间的最近邻检索逻辑
            - 处理 front camera（可同时看到双臂 gripper）与 wrist camera（只看到对应臂）的差异
            - 考虑是否需要为每个 gripper link 返回独立 mask
        """
        raise NotImplementedError(
            "_lookup_gripper_gt_mask: 需要实现 wrist camera 数据库检索。"
            "数据库应提供 action → gripper 像素 mask 的映射。"
        )

    def _match_and_interpolate_gripper(
        self,
        proj_uv: np.ndarray,
        proj_values: np.ndarray,
        gripper_gt_mask: np.ndarray,
        image_size: Tuple[int, int] = (480, 640)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        核心共享函数: 将 gripper-proj（投影稀疏点）与 gripper-gt（GT dense mask）
        进行形状匹配和插值，使 GT mask 中每个像素都获得对应的值。

        被以下函数调用:
        - _project_gripper_to_image_refine: proj_values 为颜色 [N, 3]
        - get_flow_project_refine: proj_values 为 3D flow [N, K, 3]

        流程:
            1. 构建 gripper-proj mask: 将 proj_uv 栅格化为 [H, W] binary mask
            2. 形状匹配: 将 gripper-proj mask 与 gripper-gt mask 大致对齐
               （如基于轮廓的仿射/TPS 变换、ICP 等）
               注意: 形状匹配只做一次，与 proj_values 的维度无关
            3. 插值填充: 对 gripper-gt 中每个像素，基于匹配后的 gripper-proj
               邻近点进行插值得到对应值（如 IDW、RBF、scipy.interpolate）
               插值沿 values 的所有维度进行

        Args:
            proj_uv: [N, 2] float, gripper 3D 点投影到 2D 的像素坐标 (u, v)
            proj_values: 每个投影点携带的值，支持两种形状:
                         - [N, 3]: 颜色 (来自 _project_gripper_to_image_refine)
                         - [N, K, 3]: K 帧 3D flow (来自 get_flow_project_refine)
            gripper_gt_mask: [H, W] bool, 从数据库检索的 GT gripper 像素 mask
            image_size: 图像尺寸 (H, W)

        Returns:
            result_map: gripper-gt 区域填充了插值后的值，其余为 0
                        - 当 proj_values 为 [N, 3] 时: [H, W, 3]
                        - 当 proj_values 为 [N, K, 3] 时: [H, W, K, 3]
            filled_mask: [H, W] bool, 成功填充值的像素 mask

        TODO:
            - Step 1: proj_uv 栅格化为 binary mask
            - Step 2: 实现形状匹配算法（推荐: 基于轮廓质心对齐 + 仿射变换）
            - Step 3: 实现插值方法（推荐: scipy.interpolate.griddata 或 IDW）
              对 proj_values 的所有维度统一插值
            - 处理 gripper-proj 点数过少的退化情况（直接返回零图）
            - 处理 gripper-gt mask 为空的情况
        """
        raise NotImplementedError(
            "_match_and_interpolate_gripper: 需要实现 gripper-proj 与 gripper-gt 的匹配和插值。"
            "步骤: 1) proj_uv 栅格化 2) 形状对齐 3) 逐像素插值"
        )

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        rgb: np.ndarray,
        mask_exclude: Optional[np.ndarray] = None,
        depth_range: Tuple[float, float] = (0.0, 5.0)
    ) -> np.ndarray:
        H, W = depth.shape
        
        # 调整RGB尺寸
        if rgb.shape[:2] != (H, W):
            scale_h = H / rgb.shape[0]
            scale_w = W / rgb.shape[1]
            rgb = zoom(rgb, (scale_h, scale_w, 1), order=1)
        
        # 调整mask尺寸
        if mask_exclude is not None and mask_exclude.shape[:2] != (H, W):
            mask_exclude = cv2.resize(
                mask_exclude.astype(np.uint8), (W, H), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # 生成像素坐标
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # 有效性mask
        valid_mask = (depth > depth_range[0]) & (depth < depth_range[1])
        if mask_exclude is not None:
            valid_mask = valid_mask & (~mask_exclude)
        
        # 提取有效像素
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]
        
        # 反投影
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid
        
        points_xyz = np.stack([x, y, z], axis=-1)
        colors = rgb[valid_mask].astype(np.float32) / 255.0
        
        points = np.concatenate([points_xyz, colors], axis=-1)
        
        return points

    def _transform_to_world(
        self,
        points: np.ndarray,
        extrinsic: np.ndarray
    ) -> np.ndarray:
        xyz = points[:, :3]
        colors = points[:, 3:]
        
        # DA3可能输出 (3,4)，补全为 (4,4)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        
        C2W = np.linalg.inv(extrinsic)
        
        xyz_homo = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=-1)
        xyz_world = (C2W @ xyz_homo.T).T[:, :3]
        
        return np.concatenate([xyz_world, colors], axis=-1)

    def _reconstruct_gripper_from_flow(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        C2W: np.ndarray,
        flow_2D: Dict[str, np.ndarray],
        K: int,
        depth_range: Tuple[float, float] = (0.0, 5.0)
    ) -> List[np.ndarray]:
        """
        从 wrist depth + flow mask 重建 K+1 帧 gripper 点云（世界坐标系）

        流程:
            1. mask 内像素用 depth 反投影为 T=0 3D 点（camera 坐标系）
            2. C2W 变换到世界坐标系
            3. 逐帧加 3D flow 得到 T+k 点云

        Args:
            depth: [H_d, W_d] wrist camera 深度图（DA3 输出分辨率）
            intrinsic: [3, 3] DA3 输出的内参（与 depth 分辨率匹配）
            C2W: [4, 4] camera-to-world 变换矩阵
            flow_2D: dict with 'mask' [H, W] bool, 'flow' [H, W, K, 3]
                     （image_size 分辨率，可能与 depth 不同）
            K: 未来帧数
            depth_range: 有效深度范围

        Returns:
            gripper_pts_list: List[np.ndarray], 长度 K+1, 每个 [N, 3], 世界坐标系
        """
        H_depth, W_depth = depth.shape
        mask = flow_2D['mask']    # [H_img, W_img]
        flow = flow_2D['flow']    # [H_img, W_img, K, 3]
        H_img, W_img = mask.shape

        # 将 mask 和 flow resize 到 depth 分辨率（如分辨率不同）
        if (H_depth, W_depth) != (H_img, W_img):
            mask_rs = cv2.resize(
                mask.astype(np.uint8), (W_depth, H_depth),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            flow_rs = np.zeros((H_depth, W_depth, K, 3), dtype=flow.dtype)
            for k in range(K):
                for c in range(3):
                    flow_rs[:, :, k, c] = cv2.resize(
                        flow[:, :, k, c], (W_depth, H_depth),
                        interpolation=cv2.INTER_LINEAR
                    )
        else:
            mask_rs = mask
            flow_rs = flow

        # 获取 mask 内像素坐标
        v_idx, u_idx = np.where(mask_rs)

        if len(v_idx) == 0:
            return [np.zeros((0, 3), dtype=np.float64) for _ in range(K + 1)]

        # 深度有效性过滤
        z = depth[v_idx, u_idx].astype(np.float64)
        valid = (z > depth_range[0]) & (z < depth_range[1])
        v_idx, u_idx, z = v_idx[valid], u_idx[valid], z[valid]

        if len(z) == 0:
            return [np.zeros((0, 3), dtype=np.float64) for _ in range(K + 1)]

        # 反投影为 3D 点（camera 坐标系）
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x = (u_idx.astype(np.float64) - cx) * z / fx
        y = (v_idx.astype(np.float64) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)  # [N, 3]

        # 转到世界坐标系
        pts_homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
        pts_world = (C2W @ pts_homo.T).T[:, :3]  # [N, 3]

        # T=0 + 逐帧加 flow
        gripper_pts_list = [pts_world.copy()]  # 索引 0 = T=0
        for k in range(K):
            flow_k = flow_rs[v_idx, u_idx, k, :].astype(np.float64)  # [N, 3]
            gripper_pts_list.append(pts_world + flow_k)

        return gripper_pts_list

    def get_gripper_points(
        self,
        current_future_action: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        获取当前帧+未来K帧的gripper点云（共K+1帧）

        Args:
            current_future_action: 动作序列 [K+1, 14]，包含T到T+K时刻

        Returns:
            gripper_pts_lg1_list: 左臂gripper1点云列表，长度K+1，每个 [5000, 3]，世界坐标系(front camera)
            gripper_pts_lg2_list: 左臂gripper2点云列表，长度K+1，每个 [5000, 3]，世界坐标系(front camera)
            gripper_pts_rg1_list: 右臂gripper1点云列表，长度K+1，每个 [5000, 3]，世界坐标系(front camera)
            gripper_pts_rg2_list: 右臂gripper2点云列表，长度K+1，每个 [5000, 3]，世界坐标系(front camera)
        """
        num_frames = len(current_future_action)  # K+1 帧 (当前帧T + 未来K帧)

        gripper_pts_lg1_list = []
        gripper_pts_lg2_list = []
        gripper_pts_rg1_list = []
        gripper_pts_rg2_list = []

        # 加载gripper mesh（只加载一次）
        left_gripper1_mesh = self._load_gripper_mesh("link7")
        left_gripper2_mesh = self._load_gripper_mesh("link8")
        right_gripper1_mesh = self._load_gripper_mesh("link7")
        right_gripper2_mesh = self._load_gripper_mesh("link8")

        for t in range(num_frames):  # t = 0, 1, ..., K，共K+1帧
            qpos = current_future_action[t]
            
            q_left7 = qpos[:7]
            q_right7 = qpos[-7:]
            q_left_full = q_left7.astype(np.float64)
            q_right_full = q_right7.astype(np.float64)
            
            n_joints_left_g1 = self.ets_left_gripper1.n
            n_joints_left_g2 = self.ets_left_gripper2.n
            n_joints_right_g1 = self.ets_right_gripper1.n
            n_joints_right_g2 = self.ets_right_gripper2.n
            
            T_left_gripper1_in_basel = self._fk_base_to_end(
                self.ets_left_gripper1, 
                q_left_full[:n_joints_left_g1][np.newaxis, :]
            )[0]
            T_left_gripper2_in_basel = self._fk_base_to_end(
                self.ets_left_gripper2, 
                q_left_full[:n_joints_left_g2][np.newaxis, :]
            )[0]
            T_right_gripper1_in_baser = self._fk_base_to_end(
                self.ets_right_gripper1, 
                q_right_full[:n_joints_right_g1][np.newaxis, :]
            )[0]
            T_right_gripper2_in_baser = self._fk_base_to_end(
                self.ets_right_gripper2, 
                q_right_full[:n_joints_right_g2][np.newaxis, :]
            )[0]
            
            left_gripper1_world = self._transform_gripper_to_world(
                left_gripper1_mesh,
                T_left_gripper1_in_basel @ self.T_gripper_left1,
                self.T_front2basel
            )
            left_gripper2_world = self._transform_gripper_to_world(
                left_gripper2_mesh,
                T_left_gripper2_in_basel @ self.T_gripper_left2,
                self.T_front2basel
            )
            right_gripper1_world = self._transform_gripper_to_world(
                right_gripper1_mesh,
                T_right_gripper1_in_baser @ self.T_gripper_right1,
                self.T_front2baser
            )
            right_gripper2_world = self._transform_gripper_to_world(
                right_gripper2_mesh,
                T_right_gripper2_in_baser @ self.T_gripper_right2,
                self.T_front2baser
            )
            
            gripper_pts_lg1_list.append(left_gripper1_world)
            gripper_pts_lg2_list.append(left_gripper2_world)
            gripper_pts_rg1_list.append(right_gripper1_world)
            gripper_pts_rg2_list.append(right_gripper2_world)
        
        return gripper_pts_lg1_list, gripper_pts_lg2_list, gripper_pts_rg1_list, gripper_pts_rg2_list

    def get_gripper_flow(
        self,
        gripper_pts_lg1: List[np.ndarray],
        gripper_pts_lg2: List[np.ndarray],
        gripper_pts_rg1: List[np.ndarray],
        gripper_pts_rg2: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 gripper 3D flow: 当前帧 (T=0) 点云到未来每帧的位移

        对每个 gripper link:
            flow[i, k, :] = gripper_pts[k+1][i] - gripper_pts[0][i]
        即第 i 个点从 T=0 到 T=k+1 的 3D 位移向量。

        前提: 所有帧的点云来自同一 STL 采样，点索引一一对应。

        Args:
            gripper_pts_lg1: 左臂 gripper1 点云列表, 长度 K+1, 每个 [5000, 3]
            gripper_pts_lg2: 左臂 gripper2 点云列表
            gripper_pts_rg1: 右臂 gripper1 点云列表
            gripper_pts_rg2: 右臂 gripper2 点云列表

        Returns:
            flow_lg1: [5000, K, 3] 左臂 gripper1 的 3D flow
            flow_lg2: [5000, K, 3] 左臂 gripper2 的 3D flow
            flow_rg1: [5000, K, 3] 右臂 gripper1 的 3D flow
            flow_rg2: [5000, K, 3] 右臂 gripper2 的 3D flow

        Note:
            点云索引一一对应（同一 STL 采样），因此直接做差即可。
        """
        K = len(gripper_pts_lg1) - 1  # K+1 帧中有 K 个未来帧

        def _compute_flow(pts_list):
            pts_t0 = pts_list[0]  # [5000, 3]
            return np.stack([pts_list[k + 1] - pts_t0 for k in range(K)], axis=1)  # [5000, K, 3]

        flow_lg1 = _compute_flow(gripper_pts_lg1)
        flow_lg2 = _compute_flow(gripper_pts_lg2)
        flow_rg1 = _compute_flow(gripper_pts_rg1)
        flow_rg2 = _compute_flow(gripper_pts_rg2)

        print(f"  gripper 3D flow: shape={flow_lg1.shape}, K={K}")
        return flow_lg1, flow_lg2, flow_rg1, flow_rg2

    def get_flow_project_refine(
        self,
        flow_g1: np.ndarray,
        flow_g2: np.ndarray,
        gripper_pts_g1_t0: np.ndarray,
        gripper_pts_g2_t0: np.ndarray,
        wrist_rgb: np.ndarray,
        wrist_extrinsic: np.ndarray,
        current_action: np.ndarray,
        cam_name: str = None,
        image_size: Tuple[int, int] = (480, 640)
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        将 gripper 3D flow 投影到 wrist camera 2D 像素空间，并通过 GT mask 匹配插值

        针对一只手臂的两个 gripper link (g1, g2):
            1. 将 T=0 时刻的 gripper 3D 点投影到 wrist camera 2D (gripper-proj)
            2. 从数据库检索 GT gripper mask (gripper-gt)
            3. 对每帧 flow 分别调用 _match_and_interpolate_gripper 进行匹配+插值

        与 _project_gripper_to_image_refine 的关系:
            两者共享 _lookup_gripper_gt_mask + _match_and_interpolate_gripper
            - _project_gripper_to_image_refine: 值为颜色，输出图像
            - 本函数: 值为 3D flow，输出逐像素 flow map

        Args:
            flow_g1: [5000, K, 3] gripper link1 的 3D flow
            flow_g2: [5000, K, 3] gripper link2 的 3D flow
            gripper_pts_g1_t0: [5000, 3] T=0 时刻 gripper link1 世界坐标点云
            gripper_pts_g2_t0: [5000, 3] T=0 时刻 gripper link2 世界坐标点云
            wrist_rgb: [H, W, 3] wrist camera RGB 图像（当前帧）
            wrist_extrinsic: [4, 4] wrist camera 外参 (W2C)
            current_action: [14,] 当前帧 qpos，用于数据库检索 GT mask
            cam_name: 相机名称 ('left' 或 'right')
            image_size: 图像尺寸 (H, W)

        Returns:
            flow_g1_2D: dict with:
                'mask': [H, W] bool, gripper link1 像素 mask
                'flow': [H, W, K, 3] float, 每个像素的 3D flow (非 gripper 区域为 0)
            flow_g2_2D: dict with:
                'mask': [H, W] bool, gripper link2 像素 mask
                'flow': [H, W, K, 3] float, 每个像素的 3D flow
        """
        H, W = image_size
        K = flow_g1.shape[1]
        intrinsic = self.intrinsics[cam_name]

        if wrist_extrinsic.shape == (3, 4):
            wrist_extrinsic = np.vstack([wrist_extrinsic, [0, 0, 0, 1]])

        # 检索 GT gripper mask（两个 link 共用同一 GT mask，
        # _match_and_interpolate_gripper 会基于各 link 投影位置自动区分）
        gripper_gt_mask = self._lookup_gripper_gt_mask(current_action, cam_name, image_size)

        results = []
        for link_idx, (pts_t0, flow) in enumerate([
            (gripper_pts_g1_t0, flow_g1),
            (gripper_pts_g2_t0, flow_g2)
        ]):
            # 1. 将 T=0 gripper 3D 点投影到 wrist camera 2D
            #    复用 _project_gripper_to_image_refine 的投影逻辑:
            #    extrinsic @ T_gripper2cam_mat @ world_pts
            xyz_homo = np.concatenate([pts_t0, np.ones((len(pts_t0), 1))], axis=-1)
            xyz_cam = (wrist_extrinsic @ self.T_gripper2cam_mat[cam_name] @ xyz_homo.T).T[:, :3]

            # 过滤相机后面的点
            valid_mask = xyz_cam[:, 2] > 0.01
            if valid_mask.sum() == 0:
                results.append({
                    'mask': np.zeros((H, W), dtype=bool),
                    'flow': np.zeros((H, W, K, 3), dtype=np.float32)
                })
                print(f"  {cam_name} link{link_idx+1}: 无可见投影点")
                continue

            xyz_cam_valid = xyz_cam[valid_mask]
            flow_valid = flow[valid_mask]  # [N_valid, K, 3]

            # 投影到 2D 像素坐标
            X, Y, Z = xyz_cam_valid[:, 0], xyz_cam_valid[:, 1], xyz_cam_valid[:, 2]
            u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
            v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]
            # T_gripper2cam_mat 标定导致横向左右颠倒，需翻转 u
            u = W - 1 - u
            proj_uv = np.stack([u, v], axis=-1)  # [N_valid, 2]

            # 2. 形状匹配一次 + 对所有 K 帧 flow 统一插值
            #    flow_valid: [N_valid, K, 3] → result_map: [H, W, K, 3]
            flow_map, filled_mask = self._match_and_interpolate_gripper(
                proj_uv, flow_valid, gripper_gt_mask, image_size
            )

            results.append({
                'mask': filled_mask,
                'flow': flow_map
            })
            print(f"  {cam_name} link{link_idx+1}: "
                  f"投影点 {valid_mask.sum()}, mask像素 {combined_mask.sum()}")

        return results[0], results[1]

    def _transform_gripper_to_world(
        self,
        gripper_points: np.ndarray,
        T_gripper_to_base: np.ndarray,
        T_front_to_base: np.ndarray
    ) -> np.ndarray:
        """
        将gripper点云从gripper坐标系转换到世界坐标系
        
        Args:
            gripper_points: [N, 3] gripper局部坐标系中的点
            T_gripper_to_base: [4, 4] base -> gripper的变换
            T_front_to_base: [4, 4] front -> base的变换
        
        Returns:
            points_world: [N, 3] 世界坐标系中的点
        """
        # gripper -> base -> front
        T_base_to_front = np.linalg.inv(T_front_to_base)
        T_gripper_to_front = T_base_to_front @ T_gripper_to_base
        
        # 齐次坐标变换
        gripper_points_homo = np.concatenate([gripper_points, 
                                            np.ones((len(gripper_points), 1))], axis=-1)
        points_world = (T_gripper_to_front @ gripper_points_homo.T).T[:, :3]
        
        return points_world

    def get_cond_gripper_interact(
        self, 
        DA3_pts: Tuple[np.ndarray, np.ndarray, np.ndarray],
        gripper_pts: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
        extrinsics_list: List[Dict[str, np.ndarray]],
        is_2D_aligned: bool,
        current_future_action: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        生成gripper交互条件（3D碰撞检测 + 投影到三视角）
        每个gripper独立检测碰撞并染色
        """
        scene_pts = np.concatenate([DA3_pts[0], DA3_pts[1], DA3_pts[2]], axis=0)
        scene_xyz = scene_pts[:, :3]
        kdtree = KDTree(scene_xyz)
        
        lg1_list, lg2_list, rg1_list, rg2_list = gripper_pts
        K = len(lg1_list) - 1
        
        color_collision = np.array([1.0, 0.0, 0.0])
        color_free = np.array([0.5, 0.5, 0.5])
        
        def colorize(pts, collided):
            color = color_collision if collided else color_free
            return np.concatenate([pts, np.tile(color, (len(pts), 1))], axis=-1)
        
        cond_video = []
        
        print(f"🎯 生成gripper交互条件 (K={K} 时间步, 4个gripper独立检测)")
        
        for t in range(1, K+1):
            collision_lg1 = self._check_collision(lg1_list[t], kdtree)
            collision_lg2 = self._check_collision(lg2_list[t], kdtree)
            collision_rg1 = self._check_collision(rg1_list[t], kdtree)
            collision_rg2 = self._check_collision(rg2_list[t], kdtree)
            
            all_grippers = np.concatenate([
                colorize(lg1_list[t], collision_lg1),
                colorize(lg2_list[t], collision_lg2),
                colorize(rg1_list[t], collision_rg1),
                colorize(rg2_list[t], collision_rg2),
            ], axis=0)
            
            extrinsics_t = extrinsics_list[t]
            
            frame_images = []
            for cam in self.camera_names:
                extrinsic = extrinsics_t[cam]
                if is_2D_aligned:
                    proj_image = self._project_gripper_to_image(
                        gripper_points=all_grippers,
                        cam_name=cam,
                        extrinsic=extrinsic
                    )
                else:
                    proj_image = self._project_gripper_to_image_refine(
                        gripper_points=all_grippers,
                        cam_name=cam,
                        extrinsic=extrinsic,
                        action=current_future_action[t]
                    )
                frame_images.append(proj_image)
            
            cond_video.append(frame_images)
            
            print(f"  时刻 T+{t}: "
                  f"左G1={'碰撞' if collision_lg1 else '自由'}, "
                  f"左G2={'碰撞' if collision_lg2 else '自由'}, "
                  f"右G1={'碰撞' if collision_rg1 else '自由'}, "
                  f"右G2={'碰撞' if collision_rg2 else '自由'}")
        
        return cond_video
    
    def _check_collision(
        self, 
        gripper_points: np.ndarray, 
        scene_kdtree: KDTree
    ) -> bool:

        neighbors = scene_kdtree.query_ball_point(
            gripper_points,
            r=self.collision_threshold
        )

        # 每个gripper点周围的scene点数量
        neighbor_counts = np.array([len(n) for n in neighbors])

        # 至少有若干gripper点周围scene点密集
        contact_points = neighbor_counts >= self.scene_density_threshold

        return contact_points.sum() >= self.gripper_contact_min_points

    def _project_gripper_to_image(
        self,
        gripper_points: np.ndarray,
        cam_name: str,
        extrinsic: np.ndarray,
        image_size: Tuple[int, int] = (480, 640)
    ) -> np.ndarray:
        """
        将gripper点云投影到相机图像
        """
        H, W = image_size
        intrinsic = self.intrinsics[cam_name]
        
        # 补全外参
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        
        # 转换到相机坐标系
        xyz_world = gripper_points[:, :3]
        colors = gripper_points[:, 3:6]
        
        xyz_homo = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], axis=-1)
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]
        
        # 过滤相机后面的点
        valid_mask = xyz_cam[:, 2] > 0.01
        if valid_mask.sum() == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        
        xyz_cam = xyz_cam[valid_mask]
        colors = colors[valid_mask]
        
        # 投影
        X, Y, Z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]
        
        u_px = np.round(u).astype(int)
        v_px = np.round(v).astype(int)
        
        # 过滤图像范围内的点
        in_image = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
        u_px = u_px[in_image]
        v_px = v_px[in_image]
        colors = colors[in_image]
        
        # 创建投影图像
        proj_image = np.zeros((H, W, 3), dtype=np.uint8)
        proj_image[v_px, u_px] = (colors * 255).astype(np.uint8)
        
        # 膨胀操作使投影更明显
        kernel = np.ones((3, 3), np.uint8)
        proj_image = cv2.dilate(proj_image, kernel, iterations=2)
        
        return proj_image
    
    def _project_gripper_to_image_refine(
        self,
        gripper_points: np.ndarray,
        cam_name: str,
        extrinsic: np.ndarray,
        image_size: Tuple[int, int] = (480, 640),
        action: np.ndarray = None
    ) -> np.ndarray:
        """
        将 gripper 点云投影到相机图像，通过 GT mask 匹配 + 插值获取所有 gripper 像素的颜色

        流程:
            1. 3D gripper → 相机坐标系（含校正矩阵）→ 投影到 2D (gripper-proj)
            2. 从数据库检索该 action 的 GT gripper mask (gripper-gt)
            3. 调用 _match_and_interpolate_gripper 匹配+插值，填充 GT 区域
        """
        H, W = image_size
        intrinsic = self.intrinsics[cam_name]

        # 补全外参
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        # 转换到相机坐标系（含投影校正矩阵）
        xyz_world = gripper_points[:, :3]
        colors = gripper_points[:, 3:6]

        xyz_homo = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], axis=-1)
        xyz_cam = (extrinsic @ self.T_gripper2cam_mat[cam_name] @ xyz_homo.T).T[:, :3]

        # 过滤相机后面的点
        valid_mask = xyz_cam[:, 2] > 0.01
        if valid_mask.sum() == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        xyz_cam = xyz_cam[valid_mask]
        colors = colors[valid_mask]

        # 投影到 2D 像素坐标
        X, Y, Z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]
        # T_gripper2cam_mat 标定导致横向左右颠倒，需翻转 u
        u = W - 1 - u

        # 构建 gripper-proj: 投影稀疏像素坐标 + 颜色值
        proj_uv = np.stack([u, v], axis=-1)  # [N, 2]

        # --- 匹配与插值 (共享逻辑) ---
        # 1. 从数据库检索该 action 对应的 GT gripper 像素 mask (gripper-gt)
        gripper_gt_mask = self._lookup_gripper_gt_mask(action, cam_name, image_size)
        # 2. 将 gripper-proj 与 gripper-gt 进行形状匹配 + 插值填充
        result_map, filled_mask = self._match_and_interpolate_gripper(
            proj_uv, colors, gripper_gt_mask, image_size
        )

        # 将插值结果转为图像
        proj_image = (result_map * 255).astype(np.uint8)

        return proj_image

    def get_cond_implicit(self, DA3_pts, gripper_pts):
        """隐式特征条件（暂未实现）"""
        raise NotImplementedError("implicit_3D condition not implemented yet!")

    def rgb_action2flow_cond(
        self, 
        current_obs: List[np.ndarray], 
        current_future_action: np.ndarray,
        modality: str = "3D",
        cond: str = "gripper_interact"
    ) -> Tuple[List[List[np.ndarray]], Dict[str, np.ndarray]]:
        """
        从RGB图像和动作序列生成flow条件
        
        Returns:
            cond_video: 条件视频
            arm_masks: 机械臂过滤mask字典
        """
        print(f"\n{'='*70}")
        print(f"  RGB + Action → Flow Condition (cond={cond})")
        print(f"{'='*70}\n")
        K = len(current_future_action) - 1

        # Step 1: 计算未来K帧的相机外参（用于投影）
        print(f"\nStep 1: 计算未来K帧相机外参")
        extrinsics_list = []
        for t in range(0, K + 1):
            ext_t = self.forward_kinematics(current_future_action[t])
            extrinsics_list.append(ext_t)
        print(f"  已计算 {len(extrinsics_list)} 帧外参")
        
        # Step 2: DA3推理获取第T帧的深度
        print(f"\nStep 2: DA3深度估计")
        depths, extrinsics_da3, intrinsics = self.forward_DA3(current_obs, extrinsics_list[0])
        

        if modality == "3D":
            # Step 3: 深度转点云（过滤机械臂）
            print(f"\nStep 3: 深度转点云（过滤机械臂）")
            front_pts, left_pts, right_pts, arm_masks = self.convert_depth(
                current_obs, depths, extrinsics_da3, intrinsics
            )
            
            # Step 4: 计算当前帧+未来K帧gripper点云（共K+1帧）
            print(f"\nStep 4: 计算当前帧+未来K帧gripper点云")
            gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.get_gripper_points(current_future_action)
            print(f"  总帧数: K+1={K+1} (当前帧+未来{K}帧)")
        
        if modality == "2D":
            # Step 3: 计算当前帧+未来K帧gripper点云（共K+1帧）
            print(f"\nStep 3: 计算当前帧+未来K帧gripper点云")
            gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.get_gripper_points(current_future_action) # (K+1)x5000x3
            print(f"  总帧数: K+1={K+1} (当前帧+未来{K}帧)")

            # step 4: 获取当前帧gripper点云到未来帧的3D flow
            gripper_flow_lg1, gripper_flow_lg2, gripper_flow_rg1, gripper_flow_rg2 = \
                self.get_gripper_flow(gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2) # 5000xKx3
            
            # step 5: 将3D flow投影到2D pixel，包含gripper pixel对齐与插值
            print(f"\nStep 5: 3D flow → 2D 投影 + GT mask 匹配插值")
            gripper_flow_lg1_2D, gripper_flow_lg2_2D = self.get_flow_project_refine(
                gripper_flow_lg1, gripper_flow_lg2,
                gripper_pts_lg1[0], gripper_pts_lg2[0],
                current_obs[1],
                extrinsics_da3['left'], current_future_action[0], cam_name='left')
            gripper_flow_rg1_2D, gripper_flow_rg2_2D = self.get_flow_project_refine(
                gripper_flow_rg1, gripper_flow_rg2,
                gripper_pts_rg1[0], gripper_pts_rg2[0],
                current_obs[2],
                extrinsics_da3['right'], current_future_action[0], cam_name='right')

            # Step 6: 深度转场景点云 + 重建K+1帧gripper点云
            # front基于颜色过滤，wrist基于flow mask区分场景/gripper
            # gripper用depth反投影+3D flow重建K+1帧点云（格式同3D方案）
            print(f"\nStep 6: 深度转场景点云 + 重建K+1帧gripper点云")
            front_pts, left_pts, right_pts, arm_masks, gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.convert_depth_with_flow_mask(
                    current_obs, depths, extrinsics_da3, intrinsics,
                    gripper_flow_lg1_2D, gripper_flow_lg2_2D,
                    gripper_flow_rg1_2D, gripper_flow_rg2_2D
                )
        
        # Step 5: 生成条件
        if cond == "gripper_interact":
            print(f"\nStep 5: 生成gripper交互条件")
            cond_video = self.get_cond_gripper_interact(
                (front_pts, left_pts, right_pts), 
                (gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2),
                extrinsics_list=extrinsics_list,
                is_2D_aligned=False if modality == "3D" else True,
                current_future_action=current_future_action if modality == "3D" else None
            )
            return cond_video, arm_masks
        elif cond == "implicit":
            print(f"\nStep 5: 生成隐式3D条件")
            cond_feature = self.get_cond_implicit(
                (front_pts, left_pts, right_pts), 
                (gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)
            )
            return cond_feature, arm_masks
        else:
            raise NotImplementedError(f"{cond} not implemented yet!")


# 使用示例
if __name__ == "__main__":
    # ======================== 配置参数 ========================
    h5_path = "/shared_disk/datasets/private_datasets/robot_data/agilex_data/" + \
              "clean_desk/clean_desk_T1/20250819T005_clean_desk_lsz001T2_04/episode_1.hdf5"
    
    urdf_path = "/mnt/data-2/users/wangboyuan/xxw/3d_control/robot_utils/piper_twin.urdf"
    gripper_mesh_dir = "/mnt/data-2/users/wangboyuan/xxw/3d_control/robot_utils/piper/meshes"
    model_path = "/mnt/data-2/users/wangboyuan/xxw/3d_control/Depth-Anything-3/checkpoints"
    
    # 时间参数
    target_t = 1030    # 当前时刻T
    K = 170            # 未来K个时间步 (T+1, T+2, ..., T+K)
    
    output_dir = f"./output_cond_T{target_t}_K{K}"
    os.makedirs(output_dir, exist_ok=True)
    
    # ======================== 初始化生成器 ========================
    print(f"\n{'='*70}")
    print(f"  Gripper Interaction Condition Generator")
    print(f"{'='*70}\n")
    print(f"配置:")
    print(f"  HDF5文件: {h5_path}")
    print(f"  当前时刻T: {target_t}")
    print(f"  未来步数K: {K}")
    print(f"  输出目录: {output_dir}\n")
    
    generator = CondGenerator(
        model_path=model_path,
        urdf_path=urdf_path,
        gripper_mesh_dir=gripper_mesh_dir,
        device="cuda"
    )
    
    # ======================== 从HDF5读取数据 ========================
    print(f"📂 从HDF5读取数据")
    
    with h5py.File(h5_path, "r") as f:
        # 读取第T帧的三视角RGB图像
        front_rgb = f['observations/images/cam_high'][target_t]
        left_rgb = f['observations/images/cam_left_wrist'][target_t]
        right_rgb = f['observations/images/cam_right_wrist'][target_t]
        
        current_obs = [front_rgb, left_rgb, right_rgb]
        
        # 读取从T到T+K的qpos (共K+1帧)
        current_future_action = f['observations/qpos'][target_t:target_t + K + 1]
        
        print(f"  ✓ RGB图像形状: front={front_rgb.shape}, left={left_rgb.shape}, right={right_rgb.shape}")
        print(f"  ✓ Action形状: {current_future_action.shape} (包含T到T+{K})")
    
    # ======================== 生成条件 ========================
    cond_video, arm_masks = generator.rgb_action2flow_cond(
        current_obs=current_obs,
        current_future_action=current_future_action,
        modality="3D",
        cond="gripper_interact"
    )
    
    # ======================== 保存结果 ========================
    print(f"\n💾 保存结果到 {output_dir}")
    
    # 保存输入的第T帧RGB图像
    for i, cam in enumerate(['front', 'left', 'right']):
        cv2.imwrite(
            os.path.join(output_dir, f"input_T{target_t}_{cam}.png"),
            cv2.cvtColor(current_obs[i], cv2.COLOR_RGB2BGR)
        )
    
    # 保存机械臂过滤mask
    for cam in ['front', 'left', 'right']:
        mask = arm_masks[cam]
        cam_idx = ['front', 'left', 'right'].index(cam)
        
        # 叠加可视化：被过滤区域标绿
        rgb_overlay = current_obs[cam_idx].copy()
        rgb_overlay[mask] = [0, 255, 0]
        cv2.imwrite(
            os.path.join(output_dir, f"arm_mask_overlay_{cam}.png"),
            cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)
        )
    
    # 保存生成的条件视频帧
    for k in range(K):
        for i, cam in enumerate(['front', 'left', 'right']):
            cv2.imwrite(
                os.path.join(output_dir, f"cond_T{target_t + k + 1}_{cam}.png"),
                cv2.cvtColor(cond_video[k][i], cv2.COLOR_RGB2BGR)
            )
    
    print(f"  ✓ 输入图像已保存 (T={target_t}, 3视角)")
    print(f"  ✓ 机械臂mask已保存 (3视角, 二值mask + 叠加可视化)")
    print(f"  ✓ 条件图像已保存 (T+1到T+{K}, 每时刻3视角, 共{K*3}张)")
    
    # ======================== 统计信息 ========================
    print(f"\n{'='*70}")
    print(f"✅ 完成!")
    print(f"{'='*70}")
    print(f"输出结构:")
    print(f"  cond_video: List[List[np.ndarray]]")
    print(f"    - 外层长度: {len(cond_video)} (时间步T+1到T+{K})")
    print(f"    - 内层长度: {len(cond_video[0])} (front, left, right)")
    print(f"    - 图像形状: {cond_video[0][0].shape}")
    print(f"\n使用方式:")
    print(f"  cond_video[k][cam_idx] 获取第T+k+1时刻、第cam_idx个相机的投影图像")
    print(f"  cam_idx: 0=front, 1=left, 2=right")
    print(f"{'='*70}\n")