import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
import open3d as o3d
import roboticstoolbox as rtb
import trimesh
from scipy.spatial import cKDTree


@dataclass
class CameraModel:
    name: str
    intrinsic: np.ndarray  # 3x3


class RobotFKWorldModel:
    """基于 action + URDF 的简化 gripper FK 与投影工具。"""

    def __init__(
        self,
        urdf_path: str,
        gripper_mesh_dir: str,
        link7_corr_path: Optional[str] = None,
        link8_corr_path: Optional[str] = None,
        front_to_base: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Dict[str, np.ndarray]] = None,
        default_camera: str = "right",
        h5_path: Optional[str] = None,
        preload_h5: bool = False,
        preload_sample_count: int = 6000,
    ):
        self.urdf_path = urdf_path
        self.gripper_mesh_dir = gripper_mesh_dir
        self.default_camera = default_camera
        self.h5_path = h5_path

        self.T_front2base = np.eye(4) if front_to_base is None else front_to_base.astype(np.float64)
        self.T_link7_corr = np.eye(4) if link7_corr_path is None else np.load(link7_corr_path)
        self.T_link8_corr = np.eye(4) if link8_corr_path is None else np.load(link8_corr_path)

        if camera_intrinsics is None:
            camera_intrinsics = {
                "left": np.array(
                    [[605.4948120117188, 0.0, 325.0260925292969],
                     [0.0, 605.5114135742188, 246.6322479248047],
                     [0.0, 0.0, 1.0]], dtype=np.float64
                ),
                "right": np.array(
                    [[607.4896850585938, 0.0, 332.4833984375],
                     [0.0, 606.8885498046875, 249.5357666015625],
                     [0.0, 0.0, 1.0]], dtype=np.float64
                ),
                "front": np.array(
                    [[488.615234375, 0.0, 321.0052185058594],
                     [0.0, 488.615234375, 217.4329071044922],
                     [0.0, 0.0, 1.0]], dtype=np.float64
                ),
            }
        self.cameras = {k: CameraModel(name=k, intrinsic=v.astype(np.float64)) for k, v in camera_intrinsics.items()}

        self._setup_kinematics()  # URDF 只加载一次
        self._mesh_cache: Dict[str, np.ndarray] = {}
        self._h5_cache: Dict[str, np.ndarray] = {}

        # mesh 提前缓存，后续循环不会重复加载
        self._load_mesh_points("link7", sample_count=preload_sample_count)
        self._load_mesh_points("link8", sample_count=preload_sample_count)

        # hdf5 可选提前缓存
        if preload_h5 and self.h5_path is not None:
            self.preload_hdf5(self.h5_path)

    def _setup_kinematics(self) -> None:
        links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(self.urdf_path)
        robot = rtb.Robot(
            links,
            name=name,
            manufacturer="robot",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        self.ets_link7 = robot.ets(end="link7")
        self.ets_link8 = robot.ets(end="link8")

    def preload_hdf5(self, h5_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """把常用数据一次性读到内存缓存，避免后续逐帧重复 IO。"""
        file_path = h5_path or self.h5_path
        if file_path is None:
            raise ValueError("h5_path 未提供，无法 preload")

        with h5py.File(file_path, "r") as f:
            self._h5_cache["qpos"] = f["observations/qpos"][:]
            self._h5_cache["cam_left_wrist"] = f["observations/images/cam_left_wrist"][:]
            self._h5_cache["cam_right_wrist"] = f["observations/images/cam_right_wrist"][:]
        return self._h5_cache

    @staticmethod
    def read_h5_action_and_camera_frame(
        h5_path: str,
        index: int,
        camera_key: str = "cam_right_wrist",
        qpos_key: str = "observations/qpos",
    ) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(h5_path, "r") as f:
            frame = f[f"observations/images/{camera_key}"][index]
            action = f[qpos_key][index]
        return action, frame

    def _load_mesh_points(self, link_name: str, sample_count: int = 6000) -> np.ndarray:
        cache_key = f"{link_name}_{sample_count}"
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]

        mesh_path = os.path.join(self.gripper_mesh_dir, f"{link_name}.STL")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(self.gripper_mesh_dir, f"{link_name}.stl")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"mesh 不存在: {mesh_path}")

        mesh = trimesh.load(mesh_path)
        pts, _ = trimesh.sample.sample_surface(mesh, count=sample_count)
        self._mesh_cache[cache_key] = pts.astype(np.float64)
        return self._mesh_cache[cache_key]

    def fk_gripper_pose_from_action(self, action: np.ndarray, arm: str = "right") -> Dict[str, np.ndarray]:
        action = np.asarray(action).reshape(-1)
        if action.shape[0] < 14:
            raise ValueError("action 维度不足，期望至少14(左7+右7)")

        q_arm = action[-7:] if arm == "right" else action[:7]

        n7 = self.ets_link7.n
        n8 = self.ets_link8.n
        T_base_l7 = self.ets_link7.fkine(q_arm[:n7]).A @ self.T_link7_corr
        T_base_l8 = self.ets_link8.fkine(q_arm[:n8]).A @ self.T_link8_corr

        T_world_l7 = np.linalg.inv(self.T_front2base) @ T_base_l7
        T_world_l8 = np.linalg.inv(self.T_front2base) @ T_base_l8
        return {"link7": T_world_l7, "link8": T_world_l8}

    @staticmethod
    def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        return (T @ points_h.T).T[:, :3]

    def gripper_pointcloud_from_action(
        self,
        action: np.ndarray,
        arm: str = "right",
        sample_count: int = 6000,
    ) -> Dict[str, np.ndarray]:
        poses = self.fk_gripper_pose_from_action(action, arm=arm)
        link7_local = self._load_mesh_points("link7", sample_count=sample_count)
        link8_local = self._load_mesh_points("link8", sample_count=sample_count)

        link7_world = self.transform_points(link7_local, poses["link7"])
        link8_world = self.transform_points(link8_local, poses["link8"])
        return {"link7": link7_world, "link8": link8_world, "all": np.vstack([link7_world, link8_world])}

    @staticmethod
    def extract_gripper_mask_bgr(img_bgr: np.ndarray, thresh: int = 50) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray[: gray.shape[0] // 2, :] = 255
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def project_points_to_mask(
        self,
        points_cam: np.ndarray,
        image_shape: Tuple[int, int],
        camera_name: Optional[str] = None,
        pad: int = 50,
        flip_along_h: bool = True,
        return_point_indices: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将点云投影为 mask（自适应 padding 版本）。

        说明：
        - `pad` 作为最小保底 padding；
        - 实际 padding 会根据当前投影 u/v 越界量动态计算，越界越多 padding 越大；
        - 随着 gripper 对齐后越界量变小，padding 会自动减小；
        - `flip_along_h=True` 时，会沿 H 方向做镜像翻转（上下翻转）。
        - `return_point_indices=True` 时，额外返回每个投影像素对应的原始点索引（相对 points_cam）。
        """
        camera_name = camera_name or self.default_camera
        K = self.cameras[camera_name].intrinsic

        z = points_cam[:, 2]
        valid_z = z > 1e-6
        pts = points_cam[valid_z]
        base_indices = np.flatnonzero(valid_z).astype(np.int32)

        h, w = image_shape
        if pts.shape[0] == 0:
            empty_mask = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
            empty_uv = np.zeros((0, 2), dtype=np.int32)
            if return_point_indices:
                return empty_mask, empty_uv, np.zeros((0,), dtype=np.int32)
            return empty_mask, empty_uv

        z = pts[:, 2]
        uvw = (K @ pts.T).T
        u = np.round(uvw[:, 0] / z).astype(np.int32)
        v = np.round(uvw[:, 1] / z).astype(np.int32)

        # 根据越界量自适应 padding（对称扩展，便于与 RGB/GT 居中叠加）
        left_over = max(0, int(-u.min()))
        right_over = max(0, int(u.max() - (w - 1)))
        top_over = max(0, int(-v.min()))
        bottom_over = max(0, int(v.max() - (h - 1)))
        pad_dynamic = max(pad, left_over, right_over, top_over, bottom_over)

        hp = h + 2 * pad_dynamic
        wp = w + 2 * pad_dynamic

        up = u + pad_dynamic
        vp = v + pad_dynamic

        in_canvas = (up >= 0) & (up < wp) & (vp >= 0) & (vp < hp)
        up, vp = up[in_canvas], vp[in_canvas]
        kept_indices = base_indices[in_canvas]

        mask = np.zeros((hp, wp), dtype=np.uint8)
        mask[vp, up] = 255

        if flip_along_h:
            mask = np.flip(mask, axis=0)
            vp = (hp - 1) - vp

        uv = np.stack([up, vp], axis=1)
        if return_point_indices:
            return mask, uv, kept_indices
        return mask, uv

    @staticmethod
    def overlay_masks(rgb_bgr: np.ndarray, gt_mask: np.ndarray, proj_mask: np.ndarray) -> np.ndarray:
        """
        可视化 GT 与投影 mask。

        - 当 proj_mask 为 padding 后的大图时，会自动把 rgb/gt_mask 放到同尺寸画布中心；
        - GT 以绿色绘制，投影以红色绘制。
        """
        if gt_mask.dtype != np.uint8:
            gt_mask = gt_mask.astype(np.uint8)
        if proj_mask.dtype != np.uint8:
            proj_mask = proj_mask.astype(np.uint8)

        h_rgb, w_rgb = rgb_bgr.shape[:2]
        h_proj, w_proj = proj_mask.shape[:2]

        if (h_proj, w_proj) != (h_rgb, w_rgb):
            out = np.zeros((h_proj, w_proj, 3), dtype=np.uint8)
            oy = max((h_proj - h_rgb) // 2, 0)
            ox = max((w_proj - w_rgb) // 2, 0)
            y2 = min(oy + h_rgb, h_proj)
            x2 = min(ox + w_rgb, w_proj)
            out[oy:y2, ox:x2] = rgb_bgr[: y2 - oy, : x2 - ox]

            gt_canvas = np.zeros((h_proj, w_proj), dtype=np.uint8)
            gh, gw = gt_mask.shape[:2]
            gy2 = min(oy + gh, h_proj)
            gx2 = min(ox + gw, w_proj)
            gt_canvas[oy:gy2, ox:gx2] = gt_mask[: gy2 - oy, : gx2 - ox]
            gt_mask = gt_canvas
        else:
            out = rgb_bgr.copy()

        out[gt_mask > 0] = (0, 255, 0)
        out[proj_mask > 0] = (0, 0, 255)
        return out

    @staticmethod
    def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        return rz @ ry @ rx

    def interactive_adjust_projection(
        self,
        rgb_bgr: np.ndarray,
        gripper_points_cam: np.ndarray,
        gt_gripper_mask: np.ndarray,
        init_T: Optional[np.ndarray] = None,
        camera_name: Optional[str] = None,
        step_trans: float = 0.001,
        step_rot_deg: float = 0.5,
    ) -> Tuple[np.ndarray, bool]:
        """
        6 DoF 调整:
        - 平移: w/s(+y/-y), a/d(-x/+x), q/e(+z/-z)
        - 旋转: i/k(+roll/-roll), j/l(+pitch/-pitch), u/o(+yaw/-yaw)
        - p: 保存并退出(True)
        - esc: 放弃并退出(False)
        """
        T = np.eye(4) if init_T is None else init_T.copy().astype(np.float64)
        camera_name = camera_name or self.default_camera
        step_rot = np.deg2rad(step_rot_deg)

        while True:
            pts = self.transform_points(gripper_points_cam, T)
            proj_mask, _ = self.project_points_to_mask(pts, rgb_bgr.shape[:2], camera_name)
            vis = self.overlay_masks(rgb_bgr, gt_gripper_mask, proj_mask)
            cv2.imshow("adjust_projection", vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("w"):
                T[1, 3] += step_trans
            elif key == ord("s"):
                T[1, 3] -= step_trans
            elif key == ord("a"):
                T[0, 3] -= step_trans
            elif key == ord("d"):
                T[0, 3] += step_trans
            elif key == ord("q"):
                T[2, 3] += step_trans
            elif key == ord("e"):
                T[2, 3] -= step_trans
            elif key == ord("i"):
                T[:3, :3] = self._rpy_to_rot(step_rot, 0.0, 0.0) @ T[:3, :3]
            elif key == ord("k"):
                T[:3, :3] = self._rpy_to_rot(-step_rot, 0.0, 0.0) @ T[:3, :3]
            elif key == ord("j"):
                T[:3, :3] = self._rpy_to_rot(0.0, step_rot, 0.0) @ T[:3, :3]
            elif key == ord("l"):
                T[:3, :3] = self._rpy_to_rot(0.0, -step_rot, 0.0) @ T[:3, :3]
            elif key == ord("u"):
                T[:3, :3] = self._rpy_to_rot(0.0, 0.0, step_rot) @ T[:3, :3]
            elif key == ord("o"):
                T[:3, :3] = self._rpy_to_rot(0.0, 0.0, -step_rot) @ T[:3, :3]
            elif key == ord("p"):
                cv2.destroyWindow("adjust_projection")
                return T, True
            elif key == 27:
                cv2.destroyWindow("adjust_projection")
                return init_T if init_T is not None else np.eye(4), False

    @staticmethod
    def save_action_projection_record(
        save_json: str,
        action: np.ndarray,
        projection_T: np.ndarray,
        extra: Optional[dict] = None,
    ) -> None:
        records = []
        if os.path.exists(save_json):
            with open(save_json, "r", encoding="utf-8") as f:
                records = json.load(f)

        row = {
            "action": np.asarray(action).tolist(),
            "projection_T": np.asarray(projection_T).tolist(),
        }
        if extra is not None:
            row.update(extra)

        records.append(row)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    @staticmethod
    def assign_point_index_to_gripper_mask(
        gripper_mask: np.ndarray,
        uv_pixels: np.ndarray,
    ) -> np.ndarray:
        h, w = gripper_mask.shape
        idx_map = -np.ones((h, w), dtype=np.int32)
        if uv_pixels.shape[0] == 0:
            return idx_map

        u = uv_pixels[:, 0]
        v = uv_pixels[:, 1]
        idx_map[v, u] = np.arange(len(uv_pixels), dtype=np.int32)

        ys, xs = np.where(gripper_mask > 0)
        if len(xs) == 0:
            return idx_map

        proj = uv_pixels.astype(np.float64)
        pix = np.stack([xs, ys], axis=1).astype(np.float64)
        d2 = ((pix[:, None, :] - proj[None, :, :]) ** 2).sum(axis=2)
        nn = np.argmin(d2, axis=1)
        idx_map[ys, xs] = nn.astype(np.int32)
        return idx_map

    @staticmethod
    def assign_point_index_to_gripper_mask_tree(
            gripper_mask: np.ndarray,
            uv_pixels: np.ndarray,
            point_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        For each pixel (y, x) where gripper_mask > 0,
        assign the index of the nearest projected point in uv_pixels.

        Args:
            point_indices: 可选，shape=(M,)。若提供，表示 uv_pixels 每个投影点对应的原始点索引。
                返回 idx_map 时会直接写入该原始索引。

        Returns:
            idx_map: (H, W) int32, -1 for invalid pixels
        """
        h, w = gripper_mask.shape
        idx_map = -np.ones((h, w), dtype=np.int32)

        if uv_pixels.size == 0:
            return idx_map

        # --------------------------------------------------
        # 1. 过滤 + 裁剪 uv（非常重要，防越界 + 提速）
        # --------------------------------------------------
        uv = uv_pixels.astype(np.int32, copy=False)
        u = uv[:, 0]
        v = uv[:, 1]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            return idx_map

        orig_indices = np.flatnonzero(valid).astype(np.int32)
        uv = uv[valid]

        if point_indices is None:
            point_ids = orig_indices
        else:
            point_indices = np.asarray(point_indices).reshape(-1)
            if point_indices.shape[0] != uv_pixels.shape[0]:
                raise ValueError("point_indices 长度必须与 uv_pixels 一致")
            point_ids = point_indices[valid].astype(np.int32, copy=False)

        # --------------------------------------------------
        # 2. 直接命中的像素先写入（O(M)，很快）
        # --------------------------------------------------
        # 注意：这里写入的是原始 uv_pixels 中的索引，避免因 valid 过滤导致索引重排。
        idx_map[uv[:, 1], uv[:, 0]] = point_ids

        # --------------------------------------------------
        # 3. 只对 mask 内像素做 NN 查询
        # --------------------------------------------------
        ys, xs = np.where(gripper_mask > 0)
        if xs.size == 0:
            return idx_map

        query_pixels = np.stack([xs, ys], axis=1).astype(np.float32, copy=False)
        proj_points = uv.astype(np.float32, copy=False)

        # --------------------------------------------------
        # 4. KDTree 最近邻（核心加速点）
        # --------------------------------------------------
        tree = cKDTree(proj_points)
        nn_idx = tree.query(query_pixels, k=1, workers=-1)[1].astype(np.int32)

        # nn_idx 是过滤后 uv 的局部索引，需要映射回原始 uv_pixels 索引。
        idx_map[ys, xs] = point_ids[nn_idx]
        return idx_map

    @staticmethod
    def visualize_gripper_cloud_o3d(points: np.ndarray) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1.0, 0.2, 0.2])
        o3d.visualization.draw_geometries([pcd], window_name="gripper_cloud")
