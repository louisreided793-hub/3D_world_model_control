import argparse
import json
import os
from typing import Optional

import cv2
import numpy as np

from gripper_traj import read_specific_datasets
from robot_fk_world_model import RobotFKWorldModel


def _to_bgr_if_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _load_or_init_record_book(output_json: str, h5_path: str, camera: str, arm: str) -> dict:
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            book = json.load(f)
        if not isinstance(book, dict):
            raise ValueError("现有输出文件不是 dict 结构，请删除后重试")
        if "records" not in book:
            book["records"] = {}
        return book

    return {
        "meta": {
            "h5_path": h5_path,
            "camera": camera,
            "arm": arm,
        },
        "records": {},  # key: frame_idx(str)
    }


def _save_record_book(output_json: str, book: dict) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=2)


def run_gripper_match_loop(
    h5_path: str,
    urdf_path: str,
    gripper_mesh_dir: str,
    arm: str,
    camera: str,
    output_json: str,
    sample_count: int = 6000,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_stride: int = 1,
    link7_corr_path: Optional[str] = None,
    link8_corr_path: Optional[str] = None,
    front_to_base_path: Optional[str] = None,
    init_projection_path: Optional[str] = None,
    resume: bool = True,
) -> None:
    """
    标定主流程：
    1) load RobotFKWorldModel
    2) 用 gripper_traj.py 的 read_specific_datasets 读取 hdf5
    3) for 循环逐帧标定，支持断点续标
    """
    data = read_specific_datasets(h5_path)
    if "qpos" not in data:
        raise KeyError("HDF5 中未读取到 observations/qpos")

    cam_key = "cam_right_wrist" if camera == "right" else "cam_left_wrist"
    if cam_key not in data:
        raise KeyError(f"HDF5 中未读取到 observations/images/{cam_key}")

    qpos = data["qpos"]
    frames = data[cam_key]

    T_front2base = None
    if front_to_base_path:
        T_front2base = np.load(front_to_base_path)

    model = RobotFKWorldModel(
        urdf_path=urdf_path,
        gripper_mesh_dir=gripper_mesh_dir,
        link7_corr_path=link7_corr_path,
        link8_corr_path=link8_corr_path,
        front_to_base=T_front2base,
        default_camera=camera,
        h5_path=h5_path,
        preload_h5=False,
        preload_sample_count=sample_count,
    )

    current_projection = np.eye(4)
    if init_projection_path and os.path.exists(init_projection_path):
        current_projection = np.load(init_projection_path)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    book = _load_or_init_record_book(output_json, h5_path, camera, arm)
    recorded_index = set(int(k) for k in book["records"].keys())

    total = len(qpos)
    if frame_end is None:
        frame_end = total
    frame_end = min(frame_end, total)

    print(f"[INFO] total frames: {total}, process range: [{frame_start}, {frame_end}), stride={frame_stride}")
    print("[INFO] 平移: w/a/s/d/q/e, 旋转: i/k,j/l,u/o, p保存, esc取消")
    print(f"[INFO] 断点续标 resume={'ON' if resume else 'OFF'}, 已有记录数={len(recorded_index)}")

    for idx in range(frame_start, frame_end, frame_stride):
        if resume and idx in recorded_index:
            print(f"[FRAME {idx}] skip (already recorded)")
            continue

        action = qpos[idx]
        rgb_bgr = _to_bgr_if_rgb(frames[idx])

        gripper_pts_world = model.gripper_pointcloud_from_action(
            action=action,
            arm=arm,
            sample_count=sample_count,
        )["all"]
        gt_mask = model.extract_gripper_mask_bgr(rgb_bgr)

        print(f"\n[FRAME {idx}] start interactive calibration")
        adjusted_T, accepted = model.interactive_adjust_projection(
            rgb_bgr=rgb_bgr,
            gripper_points_cam=gripper_pts_world,
            gt_gripper_mask=gt_mask,
            init_T=current_projection,
            camera_name=camera,
        )

        if accepted:
            current_projection = adjusted_T

            book["records"][str(idx)] = {
                "frame_index": int(idx),
                "action": np.asarray(action).tolist(),
                "projection_T": np.asarray(current_projection).tolist(),
                "camera": camera,
                "arm": arm,
            }
            _save_record_book(output_json, book)
            recorded_index.add(idx)
            print(f"[FRAME {idx}] saved -> {output_json}")
        else:
            print(f"[FRAME {idx}] canceled by ESC (not saved)")

    cv2.destroyAllWindows()
    print("[DONE] calibration loop finished")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gripper projection match loop")
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--urdf_path", required=True)
    parser.add_argument("--gripper_mesh_dir", required=True)
    parser.add_argument("--arm", default="right", choices=["left", "right"])
    parser.add_argument("--camera", default="right", choices=["left", "right", "front"])
    parser.add_argument("--output_json", default="./action_projection_records.json")
    parser.add_argument("--sample_count", type=int, default=6000)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--link7_corr_path", default=None)
    parser.add_argument("--link8_corr_path", default=None)
    parser.add_argument("--front_to_base_path", default=None)
    parser.add_argument("--init_projection_path", default=None)
    parser.add_argument("--no_resume", action="store_true", help="禁用断点续标")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_gripper_match_loop(
        h5_path=args.h5_path,
        urdf_path=args.urdf_path,
        gripper_mesh_dir=args.gripper_mesh_dir,
        arm=args.arm,
        camera=args.camera,
        output_json=args.output_json,
        sample_count=args.sample_count,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        link7_corr_path=args.link7_corr_path,
        link8_corr_path=args.link8_corr_path,
        front_to_base_path=args.front_to_base_path,
        init_projection_path=args.init_projection_path,
        resume=not args.no_resume,
    )
