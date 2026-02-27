import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gripper_traj import read_specific_datasets
from robot_fk_world_model import RobotFKWorldModel
from tqdm import tqdm
from time import time


def load_book(book_path: str) -> Dict:
    with open(book_path, "r", encoding="utf-8") as f:
        book = json.load(f)
    if "records" not in book:
        raise KeyError("book 中缺少 records 字段")
    return book


def _to_bgr_if_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _iter_candidates(records: Dict, arm: Optional[str] = None, camera: Optional[str] = None):
    """
    按 robot_gripper_match.py 的保存格式迭代候选：
    records[idx] = {
        "frame_index": int,
        "action": [...],
        "projection_T": [[...]],
        "camera": str,
        "arm": str,
    }
    """
    for k, rec in records.items():
        if not isinstance(rec, dict):
            continue
        if "projection_T" not in rec:
            continue

        rec_arm = rec.get("arm")
        rec_cam = rec.get("camera")
        if arm is not None and rec_arm != arm:
            continue
        if camera is not None and rec_cam != camera:
            continue

        item = dict(rec)
        if "frame_index" not in item:
            try:
                item["frame_index"] = int(k)
            except Exception:
                item["frame_index"] = None
        yield str(k), item


def get_available_arms(book: Dict, camera: str) -> List[str]:
    records = book.get("records", {})
    arms = []
    for _, rec in _iter_candidates(records, arm=None, camera=camera):
        a = rec.get("arm")
        if a in ("left", "right") and a not in arms:
            arms.append(a)
    if not arms:
        arms = ["left", "right"]
    return arms


def get_projection_T_from_book(
    book: Dict,
    action: np.ndarray,
    arm: str,
    camera: str,
    frame_index: Optional[int] = None,
) -> np.ndarray:
    records = book["records"]

    # 1) 优先按 frame_index + arm + camera 精确匹配
    if frame_index is not None:
        rec = records.get(str(frame_index), None)
        if isinstance(rec, dict) and ("projection_T" in rec):
            if rec.get("arm") == arm and rec.get("camera") == camera:
                return np.asarray(rec["projection_T"], dtype=np.float64)

    # 2) fallback: 同 arm/camera 下按 action 最近邻
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    best_d = None
    best_T = None
    for _, rec in _iter_candidates(records, arm=arm, camera=camera):
        if "action" not in rec:
            continue
        a = np.asarray(rec["action"], dtype=np.float64).reshape(-1)
        if a.shape != action.shape:
            continue
        d = float(np.linalg.norm(a - action))
        if (best_d is None) or (d < best_d):
            best_d = d
            best_T = np.asarray(rec["projection_T"], dtype=np.float64)

    if best_T is None:
        raise RuntimeError(
            f"book 中未找到可用 projection_T: arm={arm}, camera={camera}, frame_index={frame_index}"
        )
    return best_T


def normalize_mask_uv_to_rgb_canvas(
    proj_mask: np.ndarray,
    uv: np.ndarray,
    rgb_shape_hw: Tuple[int, int],
    return_valid_mask: bool = False,
):
    """将可能是 padding 尺寸的 mask/uv 映射回固定 RGB 画布尺寸。"""
    h, w = rgb_shape_hw
    hp, wp = proj_mask.shape[:2]

    out = np.zeros((h, w), dtype=np.uint8)
    uv = np.asarray(uv, dtype=np.int32)
    if uv.size == 0:
        if return_valid_mask:
            return out, np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)
        return out, np.zeros((0, 2), dtype=np.int32)

    # 以中心对齐方式做裁剪/贴图
    src_y0 = max((hp - h) // 2, 0)
    src_x0 = max((wp - w) // 2, 0)
    dst_y0 = max((h - hp) // 2, 0)
    dst_x0 = max((w - wp) // 2, 0)

    copy_h = min(h, hp)
    copy_w = min(w, wp)
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = proj_mask[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]

    # uv 同步变换
    u = uv[:, 0] - src_x0 + dst_x0
    v = uv[:, 1] - src_y0 + dst_y0
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv_new = np.stack([u[valid], v[valid]], axis=1).astype(np.int32)
    if return_valid_mask:
        return out, uv_new, valid
    return out, uv_new


def align_pts_mask_to_gt(
    pts_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """用 ECC 进行 mask 对齐，返回 aligned_mask 和 2x3 仿射矩阵。"""
    pts_f = (pts_mask > 0).astype(np.float32)
    gt_f = (gt_mask > 0).astype(np.float32)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    try:
        _, warp = cv2.findTransformECC(gt_f, pts_f, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except Exception:
        # fallback: 相位相关估计平移
        shift, _ = cv2.phaseCorrelate(gt_f, pts_f)
        warp = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)

    aligned = cv2.warpAffine(
        (pts_mask > 0).astype(np.uint8) * 255,
        warp,
        (gt_mask.shape[1], gt_mask.shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned, warp


def transform_uv_by_warp(
    uv: np.ndarray,
    warp_2x3: np.ndarray,
    canvas_hw: Tuple[int, int],
    point_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if uv.shape[0] == 0:
        return uv, point_indices
    pts = np.concatenate([uv.astype(np.float32), np.ones((uv.shape[0], 1), dtype=np.float32)], axis=1)
    trans = (warp_2x3 @ pts.T).T
    u = np.round(trans[:, 0]).astype(np.int32)
    v = np.round(trans[:, 1]).astype(np.int32)
    h, w = canvas_hw
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv_out = np.stack([u[valid], v[valid]], axis=1)
    if point_indices is None:
        return uv_out, None
    return uv_out, np.asarray(point_indices)[valid]


def save_bgr_frames_to_mp4(frames_bgr: List[np.ndarray], output_path: str, fps: int = 10) -> None:
    if len(frames_bgr) == 0:
        raise ValueError("frames 为空，无法保存视频")

    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法写入视频: {output_path}")

    for frm in frames_bgr:
        if frm.shape[:2] != (h, w):
            frm = cv2.resize(frm, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(frm)
    writer.release()


def visualize_idx_map_on_mask(
    gripper_mask: np.ndarray,
    idx_map: np.ndarray,
    uv_pixels: np.ndarray,
    bg_color=(0, 0, 0),
) -> np.ndarray:
    """
    每个 mask 像素使用其最近 projected point 的 index 着色
    """
    h, w = gripper_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = bg_color

    valid = idx_map >= 0
    if not np.any(valid):
        return vis

    # 为每个 projected point 生成一个稳定颜色
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(len(uv_pixels), 3), dtype=np.uint8)

    ys, xs = np.where(valid & (gripper_mask > 0))
    vis[ys, xs] = colors[idx_map[ys, xs]]

    # 把 projected points 画出来（白点）
    for (u, v) in uv_pixels:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(vis, (int(u), int(v)), 1, (255, 255, 255), -1)

    return vis


def build_point_value_index_map(
    idx_map: np.ndarray,
    point_indices: Optional[np.ndarray],
    point_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 idx_map(投影点索引) 映射为 point_values 原始索引图。"""
    if point_indices is None:
        mapped_idx_map = idx_map.astype(np.int64, copy=False)
    else:
        point_indices = np.asarray(point_indices).reshape(-1)
        if point_indices.ndim != 1:
            raise ValueError("point_indices 必须是一维数组")

        mapped_idx_map = -np.ones_like(idx_map, dtype=np.int64)
        valid_proj = (idx_map >= 0) & (idx_map < point_indices.shape[0])
        if np.any(valid_proj):
            mapped_idx_map[valid_proj] = point_indices[idx_map[valid_proj]]

    valid_point = (mapped_idx_map >= 0) & (mapped_idx_map < point_count)
    return mapped_idx_map, valid_point


def paint_mask_with_values(
    gripper_mask: np.ndarray,
    idx_map: np.ndarray,
    point_values: np.ndarray,
    invalid_value: float = 0.0,
    left_gripper_count: int = 5000,
    point_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按 idx_map 将点级 value 填充到像素网格，并返回填充状态和左右夹爪来源。

    Args:
        gripper_mask: (H, W) binary / uint8 mask
        idx_map:      (H, W) int, -1 表示该像素未分配到点
        point_values: (N, 3) 或 (N, K, 3)
        invalid_value: 未填充像素的默认值
        left_gripper_count: gripper_pts 前多少个点视为 left gripper
        point_indices: (M,) 可选。若提供，则 idx_map 的索引先映射到 point_values 的原始索引。
            常见于 idx_map 是 "投影后点集" 的索引，而非原始 gripper_pts 索引。

    Returns:
        result_map:
            - point_values 为 (N, 3) 时，shape=(H, W, 3)
            - point_values 为 (N, K, 3) 时，shape=(H, W, K, 3)
        filled_mask: (H, W) bool，成功填充值的位置
        lr_mask:     (H, W) int8，left=0, right=1, invalid=-1
    """
    H, W = gripper_mask.shape
    values = np.asarray(point_values)
    if values.ndim not in (2, 3) or values.shape[-1] != 3:
        raise ValueError(
            "point_values 只支持 (N, 3) 或 (N, K, 3) 两种形状，且最后一维必须为 3"
        )

    num_points = values.shape[0]
    if values.ndim == 2:
        result_map = np.full((H, W, 3), invalid_value, dtype=values.dtype)
    else:
        result_map = np.full((H, W, values.shape[1], 3), invalid_value, dtype=values.dtype)

    mapped_idx_map, valid_point = build_point_value_index_map(idx_map, point_indices, num_points)
    filled_mask = (gripper_mask > 0) & valid_point
    if np.any(filled_mask):
        ys, xs = np.where(filled_mask)
        result_map[ys, xs] = values[mapped_idx_map[ys, xs]]

    lr_mask = np.full((H, W), -1, dtype=np.int8)
    if np.any(filled_mask):
        valid_indices = mapped_idx_map[filled_mask]
        lr_mask[filled_mask] = (valid_indices >= left_gripper_count).astype(np.int8)

    return result_map, filled_mask, lr_mask


def paint_mask_with_k3_values(
    gripper_mask: np.ndarray,
    idx_map: np.ndarray,
    point_values: np.ndarray,
    invalid_value: float = 0.0,
    point_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """兼容旧接口：仅返回 (H, W, K, 3) value_map。"""
    value_map, _, _ = paint_mask_with_values(
        gripper_mask=gripper_mask,
        idx_map=idx_map,
        point_values=point_values,
        invalid_value=invalid_value,
        point_indices=point_indices,
    )
    if value_map.ndim != 4:
        raise ValueError("point_values 应为 (N, K, 3)，当前收到的是 (N, 3)")
    return value_map


def run_workflow(
    book_path: str,
    h5_path: str,
    urdf_path: str,
    gripper_mesh_dir: str,
    output_mp4: str,
    camera: str = "right",
    sample_count: int = 6000,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_stride: int = 1,
    link7_corr_path: Optional[str] = None,
    link8_corr_path: Optional[str] = None,
    front_to_base_path: Optional[str] = None,
) -> None:
    book = load_book(book_path)
    data = read_specific_datasets(h5_path)

    if "qpos" not in data:
        raise KeyError("HDF5 缺少 observations/qpos")

    cam_key = "cam_right_wrist" if camera == "right" else "cam_left_wrist"
    if cam_key not in data:
        raise KeyError(f"HDF5 缺少 observations/images/{cam_key}")

    qpos = data["qpos"]
    frames = data[cam_key]

    T_front2base = np.load(front_to_base_path) if front_to_base_path else None
    model = RobotFKWorldModel(
        urdf_path=urdf_path,
        gripper_mesh_dir=gripper_mesh_dir,
        link7_corr_path=link7_corr_path,
        link8_corr_path=link8_corr_path,
        front_to_base=T_front2base,
        default_camera=camera,
        preload_sample_count=sample_count,
    )

    total = len(qpos)
    if frame_end is None:
        frame_end = total
    frame_end = min(frame_end, total)

    vis_frames: List[np.ndarray] = []
    vis_pts_frames: List[np.ndarray] = []

    enable_mask_align = False

    for idx in tqdm(range(frame_start, frame_end, frame_stride)):
        action = qpos[idx]
        rgb_bgr = _to_bgr_if_rgb(frames[idx])
        h, w = rgb_bgr.shape[:2]

        uv_all = []
        uv_point_indices_all = []
        pts_mask_all = np.zeros((h, w), dtype=np.uint8)

        # 根据 book 中可用记录决定处理哪些 arm
        active_arms = get_available_arms(book, camera=camera)
        for arm in active_arms:
            gripper_pts = model.gripper_pointcloud_from_action(action*2, arm=arm, sample_count=sample_count)["all"]
            arm_offset = 0 if arm == "left" else sample_count
            T_proj = get_projection_T_from_book(book, action, arm=arm, camera=camera, frame_index=idx)
            gripper_pts_t = model.transform_points(gripper_pts, T_proj)

            proj_mask_pad, uv_pad, point_idx_pad = model.project_points_to_mask(
                gripper_pts_t,
                (h, w),
                camera_name=camera,
                return_point_indices=True,
            )
            proj_mask, uv, uv_valid_mask = normalize_mask_uv_to_rgb_canvas(
                proj_mask_pad,
                uv_pad,
                (h, w),
                return_valid_mask=True,
            )

            pts_mask_all = cv2.bitwise_or(pts_mask_all, proj_mask)

            if uv.shape[0] > 0:
                point_idx = point_idx_pad[uv_valid_mask] + arm_offset
                uv_all.append(uv)
                uv_point_indices_all.append(point_idx.astype(np.int32))

        if len(uv_all) == 0:
            uv_all_cat = np.zeros((0, 2), dtype=np.int32)
            uv_point_indices_cat = np.zeros((0,), dtype=np.int32)
        else:
            uv_all_cat = np.concatenate(uv_all, axis=0)
            uv_point_indices_cat = np.concatenate(uv_point_indices_all, axis=0)

        gt_mask = model.extract_gripper_mask_bgr(rgb_bgr)

        # # 关键：对齐投影 mask 到 gt_mask
        # aligned_pts_mask, warp = align_pts_mask_to_gt(pts_mask_all, gt_mask)
        # uv_aligned = transform_uv_by_warp(uv_all_cat, warp, (h, w))

        if enable_mask_align:
            # 关键：对齐投影 mask 到 gt_mask
            aligned_pts_mask, warp = align_pts_mask_to_gt(pts_mask_all, gt_mask)
            uv_final, uv_point_idx_final = transform_uv_by_warp(uv_all_cat, warp, (h, w), point_indices=uv_point_indices_cat)
        else:
            aligned_pts_mask = pts_mask_all
            uv_final = uv_all_cat
            uv_point_idx_final = uv_point_indices_cat

        # 膨胀 + 并集
        kernel = np.ones((5, 5), np.uint8)
        dilated_pts_mask = cv2.dilate(aligned_pts_mask, kernel, iterations=1)
        union_mask = cv2.bitwise_or(dilated_pts_mask, gt_mask)

        # start_time = time()
        # 将 union_mask 赋点云索引
        index_map = model.assign_point_index_to_gripper_mask_tree(
            union_mask,
            uv_final,
            point_indices=uv_point_idx_final,
        )
        # print("index select time:", time() - start_time)

        # 可视化: GT(绿) + 对齐后点云mask(红)
        vis = model.overlay_masks(rgb_bgr, gt_mask, aligned_pts_mask)
        # vis = model.overlay_masks(rgb_bgr, gt_mask, union_mask)
        cv2.putText(vis, f"idx={idx}, valid_idx_px={(index_map >= 0).sum()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        vis_pts = visualize_idx_map_on_mask(gt_mask, index_map, uv_final)
        vis_frames.append(vis)
        vis_pts_frames.append(vis_pts)

    os.makedirs(os.path.dirname(output_mp4) or ".", exist_ok=True)
    save_bgr_frames_to_mp4(vis_frames, output_mp4, fps=10)
    print(f"[DONE] saved video: {output_mp4}, frames={len(vis_frames)}")

    save_bgr_frames_to_mp4(vis_pts_frames, "pts_vis.mp4", fps=10)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project gripper point-cloud to RGB and align to GT mask")
    p.add_argument("--book_path", default="D:\\python_code\\3D_world_model\\right_arm_gripper_action_to_centroids_1d\\action_projection_records.json")
    p.add_argument("--h5_path", default="D:\\python_code\\3D_world_model\\data\\episode_1.hdf5")
    p.add_argument("--urdf_path", default="D:\\python_code\\3D_world_model\\robot_utils\\piper_twin.urdf")
    p.add_argument("--gripper_mesh_dir", default="D:\\python_code\\3D_world_model\\robot_utils\\piper\\meshes")
    p.add_argument("--output_mp4", default="test.mp4")
    p.add_argument("--camera", default="right", choices=["left", "right", "front"])
    p.add_argument("--sample_count", type=int, default=5000)
    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--frame_end", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--link7_corr_path", default=None)
    p.add_argument("--link8_corr_path", default=None)
    p.add_argument("--front_to_base_path", default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_workflow(
        book_path=args.book_path,
        h5_path=args.h5_path,
        urdf_path=args.urdf_path,
        gripper_mesh_dir=args.gripper_mesh_dir,
        output_mp4=args.output_mp4,
        camera=args.camera,
        sample_count=args.sample_count,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        link7_corr_path=args.link7_corr_path,
        link8_corr_path=args.link8_corr_path,
        front_to_base_path=args.front_to_base_path,
    )
