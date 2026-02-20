import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from griper_traj_func import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def read_specific_datasets(file_path):
    """
    读取指定数据集：base_action、cam_left_wrist、cam_right_wrist、observations/qpos
    """
    result = {}
    try:
        with h5py.File(file_path, 'r') as h5_file:
            # 1. 读取 base_action（根层级）
            if 'base_action' in h5_file:
                base_action = h5_file['base_action'][:]
                result['base_action'] = base_action
                print(f"✅ 读取 base_action: 形状 {base_action.shape}, 类型 {base_action.dtype}")
            else:
                print("❌ 未找到数据集: base_action")

            # 2. 读取 observations 组下的数据集
            obs_group_path = 'observations'
            if obs_group_path in h5_file:
                obs_group = h5_file[obs_group_path]

                # 读取 qpos（核心新增）
                if 'qpos' in obs_group:
                    qpos = obs_group['qpos'][:]
                    result['qpos'] = qpos
                    print(f"✅ 读取 observations/qpos: 形状 {qpos.shape}, 类型 {qpos.dtype}")
                else:
                    print("❌ 未找到数据集: observations/qpos")

                # 读取 effort（可选，保留）
                if 'effort' in obs_group:
                    result['effort'] = obs_group['effort'][:]
                # 读取 qvel（可选，保留）
                if 'qvel' in obs_group:
                    result['qvel'] = obs_group['qvel'][:]

            # 3. 读取图像数据集（保留原有逻辑）
            image_group_path = 'observations/images'
            if image_group_path in h5_file:
                image_group = h5_file[image_group_path]
                if 'cam_left_wrist' in image_group:
                    result['cam_left_wrist'] = image_group['cam_left_wrist'][:]
                if 'cam_right_wrist' in image_group:
                    result['cam_right_wrist'] = image_group['cam_right_wrist'][:]

    except FileNotFoundError:
        print(f"❌ 错误：文件 {file_path} 不存在")
    except Exception as e:
        print(f"❌ 读取出错：{str(e)}")

    return result


def save_frames_to_mp4(frames, output_path, fps=10):
    """
    将图像帧数组保存为 MP4 视频
    :param frames: 图像帧数组，形状为 (帧数, 高度, 宽度, 通道数)
    :param output_path: 输出MP4文件路径（如 'cam_left_wrist.mp4'）
    :param fps: 视频帧率（默认10帧/秒）
    """
    # 检查输入帧是否为空
    if frames.size == 0:
        print("❌ 图像帧数据为空，无法生成视频")
        return

    # 获取帧的尺寸（高度、宽度）和通道数
    height, width, channels = frames.shape[1], frames.shape[2], frames.shape[3]

    # 定义视频编码器（MP4格式）
    # mp4v 是兼容大部分播放器的编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 创建VideoWriter对象，指定输出路径、编码器、帧率、帧尺寸
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)  # opencv中尺寸是 (宽度, 高度)，注意顺序
    )

    if not video_writer.isOpened():
        print("❌ 无法创建视频写入对象，请检查输出路径或编码器")
        return

    # 逐帧写入视频
    print(f"📽️ 开始生成视频，共 {len(frames)} 帧，帧率 {fps} FPS...")
    for i, frame in enumerate(frames):
        # 注意：h5中的图像是 RGB 格式，而OpenCV默认处理 BGR 格式，需要转换
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        # 打印进度（每50帧打印一次）
        if (i + 1) % 50 == 0:
            print(f"进度：{i + 1}/{len(frames)} 帧已写入")

    # 释放视频写入对象
    video_writer.release()
    print(f"✅ 视频已保存至：{output_path}")


def visualize_base_action(base_action_data, save_fig=False, fig_path="base_action_visualization.png"):
    """
    可视化 base_action 数据（500,14）
    :param base_action_data: 输入的 base_action 数组，形状 (500,14)
    :param save_fig: 是否保存图片（默认不保存）
    :param fig_path: 保存图片的路径
    """

    # 提取两列数据，命名为维度1、维度2（可根据实际含义修改）
    x_axis = np.arange(500)  # 帧序号（0-499）
    dim1 = base_action_data[:, 6]  # 第一列数据
    dim2 = base_action_data[:, 13]  # 第二列数据

    # 创建画布，设置大小
    plt.figure(figsize=(12, 8))

    # ========== 子图1：折线图（查看时序变化） ==========
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, dim1, label='base_action 维度1', color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, dim2, label='base_action 维度2', color='orange', linewidth=1.5, alpha=0.8)
    plt.title('base_action 时序变化（500帧）', fontsize=14, fontweight='bold')
    plt.xlabel('帧序号', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.grid(True, alpha=0.3)  # 显示网格
    plt.legend(loc='best')  # 显示图例

    # ========== 子图2：散点图（查看两维度相关性） ==========
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(dim1, dim2, c=x_axis, cmap='viridis', alpha=0.7, s=10)
    plt.title('base_action 维度1 vs 维度2（颜色表示帧序号）', fontsize=14, fontweight='bold')
    plt.xlabel('维度1', fontsize=12)
    plt.ylabel('维度2', fontsize=12)
    plt.grid(True, alpha=0.3)
    # 添加颜色条（表示帧序号）
    cbar = plt.colorbar(scatter)
    cbar.set_label('帧序号', fontsize=10)

    # 调整子图间距
    plt.tight_layout()

    # 保存图片（可选）
    if save_fig:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图片已保存至：{fig_path}")

    # 显示图像
    plt.show()


def process_gripper_image(img, is_save=False, save_name=None):
    # 1. 读取图像
    # img = cv2.imread(image_path)
    # if img is None:
    #     print("无法读取图像，请检查路径")
    #     return
    camera_frame_list = []
    centroid_contour_map = {}

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # add 形态处理
    upper_part_height = gray.shape[0] // 2
    gray[:upper_part_height, :] = 255

    # 3. 二值化：提取黑色夹爪（阈值可根据实际图像调整）
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # 4. 形态学操作：去除噪点，增强夹爪轮廓
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 筛选有效轮廓（面积大于阈值，避免噪点）
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # 面积阈值，可根据图像分辨率调整
            valid_contours.append(cnt)

    # 7. 初始化可视化图像
    vis_img = img.copy()

    # 8. 遍历每个有效轮廓（左右夹爪）
    for i, cnt in enumerate(valid_contours):
        # 计算质心
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # 计算凸包
        hull = cv2.convexHull(cnt)

        # 增加计算
        # cnt的形状是 (N, 1, 2)，先reshape为 (N, 2) 便于计算
        cnt_points = cnt.reshape(-1, 2)  # 轮廓点数组：[[x1,y1], [x2,y2], ...]
        # 计算每个点相对于质心的偏移：相对x = 点x - 质心x，相对y = 点y - 质心y
        relative_points = cnt_points - np.array([cX, cY])  # 形状 (N, 2)

        # -------- 存储质心与轮廓的映射关系 --------
        centroid_contour_map["Centroid_" + str(i+1)] = {
            "contour": cnt,  # 原始轮廓点
            "relative_points": relative_points,  # 相对质心的坐标
            "hull": hull,  # 凸包
            "area": cv2.contourArea(cnt),  # 轮廓面积（可选）,
            "center_point": np.array([cX, cY])
        }

        # 可视化：绘制轮廓、凸包和质心
        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 2)  # 绿色轮廓
        cv2.drawContours(vis_img, [hull], -1, (255, 0, 0), 2)  # 蓝色凸包
        cv2.circle(vis_img, (cX, cY), 5, (0, 0, 255), -1)       # 红色质心
        cv2.putText(vis_img, f"Centroid {i+1}: ({cX}, {cY})", (cX - 50, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # # 9. 显示结果
    # cv2.imshow("Gripper Mask & Visualization", vis_img)
    # cv2.imshow("Binary Mask", thresh)
    #
    # # 等待按键后关闭窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if is_save:
        save_centroid_contour_map_to_json(centroid_contour_map, save_name + "_centroid_contour_map.json")

    # save_frames_to_mp4(np.stack(camera_frame_list), "test.mp4", fps=10)
    return centroid_contour_map, vis_img


def draw_mask_by_centroid_and_relative(
        cX, cY,
        relative_points,
        img_size,
        save_mask=False,
        save_path="generated_mask.png"
):
    """
    基于质心和相对坐标绘制mask
    :param cX: 质心X坐标（int）
    :param cY: 质心Y坐标（int）
    :param relative_points: 轮廓点相对质心的偏移数组，形状 (N, 2)，如 [[x1_rel, y1_rel], [x2_rel, y2_rel], ...]
    :param img_size: 图像尺寸 (height, width)，如 (480, 640)
    :param save_mask: 是否保存mask（默认False）
    :param save_path: mask保存路径（默认generated_mask.png）
    :return: 生成的mask图像（单通道，0=背景，255=目标区域）
    """
    # 1. 输入校验
    if relative_points.ndim != 2 or relative_points.shape[1] != 2:
        print("❌ relative_points格式错误，需为 (N, 2) 的二维数组")
        return np.zeros(img_size, dtype=np.uint8)
    if not isinstance(cX, int) or not isinstance(cY, int):
        print("❌ 质心坐标需为整数")
        cX, cY = int(cX), int(cY)  # 兼容浮点数质心

    # 2. 还原完整轮廓点：相对坐标 + 质心坐标
    # relative_points是 [x_rel, y_rel]，完整坐标 = [cX + x_rel, cY + y_rel]
    contour_points = relative_points + np.array([cX, cY], dtype=np.int32)
    # 转换为OpenCV要求的轮廓格式：(N, 1, 2)，且数据类型为int32
    contour = contour_points.reshape(-1, 1, 2).astype(np.int32)

    # 3. 初始化全黑mask
    mask = np.zeros(img_size, dtype=np.uint8)

    # 4. 绘制填充的轮廓（生成mask）
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 5. 可选：保存mask
    # if save_mask:
    #     cv2.imwrite(save_path, mask)
    #     print(f"✅ mask已保存至：{save_path}")

    # 6. 可视化mask
    # cv2.imshow(f"Mask (Centroid: ({cX}, {cY}))", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask


def save_centroid_contour_map_to_json(centroid_contour_map, save_path):
    """
    将centroid_contour_map保存为JSON文件
    :param centroid_contour_map: 质心-轮廓映射字典
    :param save_path: JSON保存路径（如 "centroid_contour_map.json"）
    """
    # 定义序列化字典（将不可序列化的对象转为列表）
    serializable_dict = {}

    for key, info in centroid_contour_map.items():
        # 质心转为字符串作为JSON的key（JSON不支持元组key）
        centroid_key = key
        serializable_dict[centroid_key] = {
            # 轮廓/凸包：转为列表（OpenCV轮廓是 (N,1,2) 的数组）
            "contour": info["contour"].reshape(-1, 2).tolist(),
            "relative_points": info["relative_points"].tolist(),
            "hull": info["hull"].reshape(-1, 2).tolist(),
            "area": float(info["area"])  # 确保浮点数序列化
        }

    # 保存为JSON
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f, indent=4)
        print(f"✅ centroid_contour_map已保存至：{save_path}")
    except Exception as e:
        print(f"❌ 保存JSON失败：{str(e)}")


def load_centroid_contour_map_from_json(load_path):
    """
    从JSON文件加载centroid_contour_map，并还原为原格式
    :param load_path: JSON文件路径
    :return: 还原后的centroid_contour_map（与原格式一致）
    """
    if not os.path.exists(load_path):
        print(f"❌ 文件不存在：{load_path}")
        return None

    try:
        with open(load_path, "r", encoding="utf-8") as f:
            serializable_dict = json.load(f)
    except Exception as e:
        print(f"❌ 读取JSON失败：{str(e)}")
        return None

    # 还原为原格式的centroid_contour_map
    centroid_contour_map = {}
    for centroid_key, info in serializable_dict.items():

        # 还原NumPy数组 + OpenCV轮廓格式
        # 轮廓：列表 → NumPy数组 → (N,1,2) int32格式（OpenCV要求）
        contour = np.array(info["contour"], dtype=np.int32).reshape(-1, 1, 2)
        # 相对坐标：列表 → NumPy数组
        relative_points = np.array(info["relative_points"], dtype=np.int32)
        # 凸包：列表 → NumPy数组 → (N,1,2) int32格式
        hull = np.array(info["hull"], dtype=np.int32).reshape(-1, 1, 2)
        # 面积：还原为浮点数
        area = float(info["area"])

        # 构建原格式的字典
        centroid_contour_map[centroid_key] = {
            "contour": contour,
            "relative_points": relative_points,
            "hull": hull,
            "area": area
        }

    print(f"✅ 成功从 {load_path} 加载centroid_contour_map，共{len(centroid_contour_map)}个质心")
    return centroid_contour_map


def overlay_mask_on_rgb(rgb, mask, color=(0, 255, 0), alpha=0.4):
    """
    rgb:  HxWx3 (uint8)
    mask: HxW   (0/1 或 0/255)
    color: 叠加颜色 (B,G,R) for OpenCV
    alpha: 透明度
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # 统一 mask 为 0/255
    if mask.max() == 1:
        mask = mask * 255

    # 创建彩色 mask
    color_mask = np.zeros_like(rgb, dtype=np.uint8)
    color_mask[mask > 0] = color

    # alpha 融合
    overlay = cv2.addWeighted(rgb, 1.0, color_mask, alpha, 0)

    return overlay


def _to_jsonable(obj):
    """把 numpy 类型递归转成 Python 原生类型，便于 json.dump。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def load_records(save_dir="./episode_1_centroids"):
    with open(os.path.join(save_dir, "records.json"), "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


def find_nearest_gripper_center(records, target_cX, target_cY, which="both"):
    """
    which: "left" / "right" / "both"
    返回 dict:
      {
        "which": "left" or "right",
        "index": int,
        "action": float,
        "center": [x,y],
        "distance": float,
        "centroid_contour_map": {...}
      }
    """
    tgt = np.array([float(target_cX), float(target_cY)], dtype=np.float64)

    best = None

    for r in records:
        candidates = []
        if which in ("left", "both"):
            candidates.append(("left", np.array(r["left_gripper_center"], dtype=np.float64)))
        if which in ("right", "both"):
            candidates.append(("right", np.array(r["right_gripper_center"], dtype=np.float64)))

        for side, c in candidates:
            d = float(np.linalg.norm(c - tgt))
            if (best is None) or (d < best["distance"]):
                best = {
                    "which": side,
                    "index": int(r["index"]),
                    "action": float(r["action"]),
                    "center": [float(c[0]), float(c[1])],
                    "distance": d,
                    "centroid_contour_map": r["centroid_contour_map"],
                }

    if best is None:
        raise RuntimeError("No valid records found for search (records empty or missing centers).")

    return best

def find_nearest_gripper_center_weighted(records, target_cX, target_cY, which="both", wx=1.0, wy=1.0):
    tgt = np.array([float(target_cX), float(target_cY)], dtype=np.float64)
    W = np.array([wx, wy], dtype=np.float64)  # 等价于对 (dx,dy) 做缩放

    best = None
    for r in records:
        if which in ("left", "both"):
            c = np.array(r["left_gripper_center"], dtype=np.float64)
            d = (c - tgt)
            d2 = float(np.sum((d**2) * W))
            if (best is None) or (d2 < best["distance2"]):
                best = {"which":"left","index":int(r["index"]), "action":float(r["action"]),
                        "center":[float(c[0]), float(c[1])], "distance2":d2,
                        "centroid_contour_map": r["centroid_contour_map"]}

        if which in ("right", "both"):
            c = np.array(r["right_gripper_center"], dtype=np.float64)
            d = (c - tgt)
            d2 = float(np.sum((d**2) * W))
            if (best is None) or (d2 < best["distance2"]):
                best = {"which":"right","index":int(r["index"]), "action":float(r["action"]),
                        "center":[float(c[0]), float(c[1])], "distance2":d2,
                        "centroid_contour_map": r["centroid_contour_map"]}

    if best is None:
        raise RuntimeError("No valid records found for search.")

    best["distance"] = float(np.sqrt(best["distance2"]))
    return best


def get_map_by_action_nearest(records, action_value):
    a = float(action_value)

    actions = np.array([float(r["action"]) for r in records], dtype=np.float64)

    idx = int(np.argmin(np.abs(actions - a)))

    r = records[idx]
    return {
        "index": r["index"],
        "action": float(r["action"]),
        "centroid_contour_map": r["centroid_contour_map"],
        "action_error": float(abs(actions[idx] - a))
    }


def refine_mask_by_local_threshold(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    near_radius: int = 30,
    upper_half_white: bool = True,
    thr: int = 50,
    close_open_kernel: int = 5,
    use_intersection: bool = True,   # 新增开关
) -> dict:
    """
    输入:
      img_bgr: (H,W,3) BGR 图
      mask:    (H,W)   0/255 的检索轮廓mask

    参数:
      use_intersection:
          True  -> refined = mask ∩ thresh_near
          False -> refined = thresh_near (完全重新预测)

    输出:
      dict 包含 refined_mask / near_region / thresh / thresh_near
    """

    H, W = img_bgr.shape[:2]

    # --- 0) 保证 mask 为 0/255 uint8 ---
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # --- 1) 生成 mask 附近区域 ---
    k = 2 * near_radius + 1
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near_region = cv2.dilate(mask_u8, near_kernel, iterations=1)

    # --- 2) 灰度化 ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).copy()
    if upper_half_white:
        gray[:H // 2, :] = 255

    # --- 3) 阈值提取黑色 ---
    _, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    # --- 4) 形态学 ---
    kernel = np.ones((close_open_kernel, close_open_kernel), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # --- 5) ROI 约束 ---
    thresh_near = cv2.bitwise_and(thresh, near_region)

    # --- 6) 输出策略 ---
    if use_intersection:
        refined = cv2.bitwise_and(mask_u8, thresh_near)
    else:
        refined = thresh_near

    return {
        "refined_mask": refined,
        "near_region": near_region,
        "thresh": thresh,
        "thresh_near": thresh_near,
        "use_intersection": use_intersection,
    }


# ==================== 调用示例 ====================
if __name__ == "__main__":
    # 替换为你的 .h5 文件路径
    h5_file_path = "data/episode_1.hdf5"
    mp4_output_path = "cam_right_wrist_output_mask.mp4"

    # d = np.load("trans_mat/right_trans_mat.npy")
    # print(d)
    # exit()

    # 读取指定数据集
    specific_data = read_specific_datasets(h5_file_path)
    qpos = specific_data['qpos']

    cam_wrist_frames = specific_data['cam_right_wrist']

    # 保存为MP4，可调整帧率（如改为20）
    # save_frames_to_mp4(cam_left_wrist_frames, mp4_output_path, fps=10)
    # 可视化action动作
    # visualize_base_action(
    #     base_action_data=qpos,
    #     save_fig=True,  # 改为 True 可保存图片
    #     fig_path="base_action_plot.png"
    # )

    # 验证代码 -------------------------------
    # gripper_action = qpos[:, 6:7]
    gripper_action = qpos[:, 13:14]
    left_arm_left_gripper, left_arm_right_gripper, meta = load_models("./right_arm_gripper_action_to_centroids_1d")
    # load json
    # loaded_map = load_centroid_contour_map_from_json("centroid_contour_map.json")
    # pred_c1, pred_c2 = predict_from_loaded_models(
    #     left_arm_left_gripper, left_arm_right_gripper, action_rand
    # )

    camera_frame_list = []
    records = load_records("./right_arm_gripper_action_to_centroids_1d")
    for index, cam_frame_raw in enumerate(cam_wrist_frames):
        action = gripper_action[index]
        cam_frame = cam_frame_raw.copy()
        pred_list = predict_from_loaded_models(
            left_arm_left_gripper, left_arm_right_gripper, action
        )
        ret = get_map_by_action_nearest(records, action)
        # print(ret["action_error"])
        # ret = records[index]
        centroid_contour_map = ret["centroid_contour_map"]
        for key, map_dict in centroid_contour_map.items():
            contour_points = np.asarray(map_dict["contour"])
            contour = contour_points.reshape(-1, 1, 2).astype(np.int32)

            # 3. 初始化全黑mask
            mask = np.zeros(cam_frame.shape[:2], dtype=np.uint8)

            # 4. 绘制填充的轮廓（生成mask）
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            out = refine_mask_by_local_threshold(
                img_bgr=cam_frame,  # 你的原图
                mask=mask,
                near_radius=25,  # 可调：8~20 常用
                upper_half_white=True,
                thr=50,
                close_open_kernel=5,
                use_intersection=False
            )
            mask_refined = out["refined_mask"]
            cam_frame = overlay_mask_on_rgb(cam_frame, mask_refined, color=(0, 255, 0), alpha=0.4)
        # for pred_index, pred_one in enumerate(pred_list):
        #     target_cX, target_cY = pred_one
        #
        #     ret = find_nearest_gripper_center(records, target_cX, target_cY, which="both")
        #
        #     centroid_contour_map = ret["centroid_contour_map"]
        #
        #     print("Nearest side:", ret["which"])
        #     print("Nearest index:", ret["index"])
        #     print("Action:", ret["action"])
        #     print("Center:", ret["center"])
        #     print("Distance:", ret["distance"])
        #
        #     # 你要的 centroid_contour_map：
        #     centroid_contour_map = ret["centroid_contour_map"]
        #     relative_points = centroid_contour_map[list(centroid_contour_map.keys())[pred_index]]["relative_points"]
        #
        #     # relative_points = loaded_map[list(loaded_map.keys())[pred_index]]["relative_points"]
        #
        #     mask = draw_mask_by_centroid_and_relative(
        #         cX=target_cX,
        #         cY=target_cY,
        #         relative_points=np.asarray(relative_points),
        #         img_size=cam_frame.shape[:2],
        #         save_mask=True
        #     )
        #     cam_frame = overlay_mask_on_rgb(cam_frame, mask, color=(0, 255, 0), alpha=0.4)
        camera_frame_list.append(cam_frame)
        print(cam_frame.shape)

    camera_frame_list = np.stack(camera_frame_list)
    print(camera_frame_list.shape)
    save_frames_to_mp4(camera_frame_list, mp4_output_path, fps=10)

    exit()

    # cam_frame = cam_wrist_frames[0]
    # gripper_pose_dict = {}
    # records = []  # ✅ 用于集中保存的列表
    # camera_frame_list = []
    # for index, cam_frame in enumerate(cam_wrist_frames):
    #     if index == 0:
    #         process_gripper_image(cam_frame, is_save=True, save_name="episode_1")
    #     centroid_contour_map, vis_image = process_gripper_image(cam_frame)
    #     gripper_pose_dict[index] = {}
    #     gripper_pose_dict[index]["action"] = gripper_action[index]
    #     gripper_pose_dict[index]["left_gripper_center"] = centroid_contour_map["Centroid_1"]["center_point"]
    #     gripper_pose_dict[index]["right_gripper_center"] = centroid_contour_map["Centroid_2"]["center_point"]
    #     gripper_pose_dict[index]["centroid_contour_map"] = centroid_contour_map
    #
    #     rec = {
    #         "index": index,
    #         "action": float(gripper_action[index]),
    #         "left_gripper_center": gripper_pose_dict[index]["left_gripper_center"],
    #         "right_gripper_center": gripper_pose_dict[index]["right_gripper_center"],
    #         "centroid_contour_map": centroid_contour_map,
    #     }
    #     records.append(_to_jsonable(rec))
    #     camera_frame_list.append(vis_image)
    # save_frames_to_mp4(np.stack(camera_frame_list), "test.mp4", fps=10)
    #
    #
    # # ========= 落盘保存 =========
    # save_dir = "./right_arm_gripper_action_to_centroids_1d"
    # os.makedirs(save_dir, exist_ok=True)
    #
    # # 1) 保存完整 records（推荐）
    # with open(os.path.join(save_dir, "records.json"), "w", encoding="utf-8") as f:
    #     json.dump(records, f, indent=2, ensure_ascii=False)
    #
    # # 2) 同时单独保存 left/right center 便于快速加载
    # left_centers = [r["left_gripper_center"] for r in records]
    # right_centers = [r["right_gripper_center"] for r in records]
    #
    # np.save(os.path.join(save_dir, "left_centers.npy"), np.asarray(left_centers, dtype=np.float32))  # (N,2)
    # np.save(os.path.join(save_dir, "right_centers.npy"), np.asarray(right_centers, dtype=np.float32))  # (N,2)


    A, Y1, Y2 = build_1d_dataset(gripper_pose_dict)
    model_c1, model_c2 = fit_action_to_centroids_1d(
        A, Y1, Y2,
        poly_degree=5,   # 一维输入一般 3~7 都行；5 通常足够灵活又稳定
        ridge_alpha=1e-2
    )

    save_dir = "./right_arm_gripper_action_to_centroids_1d"
    p1, p2 = save_models(save_dir, model_c1, model_c2, A)
    left_arm_left_gripper, left_arm_right_gripper, meta = load_models("./right_arm_gripper_action_to_centroids_1d")
    print("Meta info:", meta)

    a_min = meta["action_min"]
    a_max = meta["action_max"]

    action_rand = float(np.random.uniform(a_min, a_max))

