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
) -> Tuple[np.ndarray, np.ndarray]:
    """将可能是 padding 尺寸的 mask/uv 映射回固定 RGB 画布尺寸。"""
    h, w = rgb_shape_hw
    hp, wp = proj_mask.shape[:2]

    out = np.zeros((h, w), dtype=np.uint8)
    uv = np.asarray(uv, dtype=np.int32)
    if uv.size == 0:
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


def transform_uv_by_warp(uv: np.ndarray, warp_2x3: np.ndarray, canvas_hw: Tuple[int, int]) -> np.ndarray:
    if uv.shape[0] == 0:
        return uv
    pts = np.concatenate([uv.astype(np.float32), np.ones((uv.shape[0], 1), dtype=np.float32)], axis=1)
    trans = (warp_2x3 @ pts.T).T
    u = np.round(trans[:, 0]).astype(np.int32)
    v = np.round(trans[:, 1]).astype(np.int32)
    h, w = canvas_hw
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.stack([u[valid], v[valid]], axis=1)


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


def paint_mask_with_k3_values(
    gripper_mask: np.ndarray,
    idx_map: np.ndarray,
    point_values: np.ndarray,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """
    Assign [k,3] values to each pixel in gripper_mask using idx_map.

    Args:
        gripper_mask: (H, W) binary mask
        idx_map:      (H, W) int, -1 means invalid
        point_values: (N, k, 3)

    Returns:
        value_map: (H, W, k, 3)
    """
    H, W = gripper_mask.shape
    k = point_values.shape[1]

    value_map = np.full((H, W, k, 3), invalid_value, dtype=point_values.dtype)

    valid = (gripper_mask > 0) & (idx_map >= 0)
    if not np.any(valid):
        return value_map

    ys, xs = np.where(valid)
    value_map[ys, xs] = point_values[idx_map[ys, xs]]

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
        pts_mask_all = np.zeros((h, w), dtype=np.uint8)

        # 根据 book 中可用记录决定处理哪些 arm
        active_arms = get_available_arms(book, camera=camera)
        for arm in active_arms:
            gripper_pts = model.gripper_pointcloud_from_action(action*2, arm=arm, sample_count=sample_count)["all"]
            T_proj = get_projection_T_from_book(book, action, arm=arm, camera=camera, frame_index=idx)
            gripper_pts_t = model.transform_points(gripper_pts, T_proj)

            proj_mask_pad, uv_pad = model.project_points_to_mask(
                gripper_pts_t,
                (h, w),
                camera_name=camera,
            )
            proj_mask, uv = normalize_mask_uv_to_rgb_canvas(proj_mask_pad, uv_pad, (h, w))

            pts_mask_all = cv2.bitwise_or(pts_mask_all, proj_mask)

            if uv.shape[0] > 0:
                uv_all.append(uv)

        if len(uv_all) == 0:
            uv_all_cat = np.zeros((0, 2), dtype=np.int32)
        else:
            uv_all_cat = np.concatenate(uv_all, axis=0)

        gt_mask = model.extract_gripper_mask_bgr(rgb_bgr)

        # # 关键：对齐投影 mask 到 gt_mask
        # aligned_pts_mask, warp = align_pts_mask_to_gt(pts_mask_all, gt_mask)
        # uv_aligned = transform_uv_by_warp(uv_all_cat, warp, (h, w))

        if enable_mask_align:
            # 关键：对齐投影 mask 到 gt_mask
            aligned_pts_mask, warp = align_pts_mask_to_gt(pts_mask_all, gt_mask)
            uv_final = transform_uv_by_warp(uv_all_cat, warp, (h, w))
        else:
            aligned_pts_mask = pts_mask_all
            uv_final = uv_all_cat

        # 膨胀 + 并集
        kernel = np.ones((5, 5), np.uint8)
        dilated_pts_mask = cv2.dilate(aligned_pts_mask, kernel, iterations=1)
        union_mask = cv2.bitwise_or(dilated_pts_mask, gt_mask)

        # start_time = time()
        # 将 union_mask 赋点云索引
        index_map = model.assign_point_index_to_gripper_mask_tree(union_mask, uv_final)
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
