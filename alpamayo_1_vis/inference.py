# -*- coding: utf-8 -*-

"""
End‑to‑end demo for Alpamayo‑R1 trajectory prediction and visualization.
"""

from __future__ import annotations
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from torch import Tensor

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from dataset import PhysicalAIAVDatasetInterface

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    clip_id: str = "eed514a0-a366-4550-b9bd-4c296c531511"
    t0_us: int = 10_000_000  
    num_history_steps: int = 16
    num_future_steps: int = 64
    time_step: float = 0.1 
    num_frames: int = 4  

    extrinsics_parquet: Path = Path(
        "/mnt/alpamayo/datas/calibration/sensor_extrinsics"
        "/sensor_extrinsics.chunk_0000.parquet"
    )
    intrinsics_parquet: Path = Path(
        "/mnt/alpamayo/datas/calibration/camera_intrinsics"
        "/camera_intrinsics.chunk_0000.parquet"
    )
    model_dir: Path = Path("/mnt/models/r1_10B")
    video_root: Path = Path("/mnt/alpamayo/datas/camera")
    output_image: Path = Path("frontwide_with_trajectory.jpg")
    extracted_frame: Path = Path("frame_10s.jpg")

    verbose: bool = True
    device: str = "cuda"  
    dtype: torch.dtype = torch.bfloat16

def extract_frame_cv2(video_path, timestamp_us,output_path,*,verbose = True,) :
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Unable to retrieve FPS from video; check the file.")

    if verbose:
        logger.info("Video FPS: %.3f", fps)

    target_ms = timestamp_us / 1_000.0  
    if verbose:
        logger.info("Target timestamp: %d µs (%.3f ms)", timestamp_us, target_ms)

    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
    ret, frame = cap.read()
    if not ret:
        frame_idx = int(timestamp_us / 1_000_000 * fps)
        if verbose:
            logger.warning(
                "Time‑based seek failed, falling back to frame index %d", frame_idx
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame at timestamp {timestamp_us} µs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write image to {output_path}")

    if verbose:
        logger.info("Frame saved to %s", output_path)


def quat_to_rot(x, y, z, w):
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Zero‑norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def world_to_cam(pt_world, R_wc, t_wc):
    return R_wc.T @ (pt_world - t_wc)


def project_fisheye( pt_cam, fx_poly, cx, cy,):
    X, Y, Z = pt_cam
    if Z <= 0:
        return None
    r_xy = np.hypot(X, Y)  
    theta = np.arctan2(r_xy, Z) 
    radius = np.polyval(fx_poly, theta)

    if r_xy == 0:
        u, v = cx, cy
    else:
        u = cx + radius * (X / r_xy)
        v = cy + radius * (Y / r_xy)

    return np.array([u, v], dtype=np.float64)

def load_extrinsics(parquet_path, clip_id, sensor_name) :
    df = pd.read_parquet(parquet_path)
    row = df.loc[(clip_id, sensor_name)]
    qx, qy, qz, qw = row[["qx", "qy", "qz", "qw"]].astype(float).values
    tx, ty, tz = row[["x", "y", "z"]].astype(float).values
    logger.info("Extrinsics for %s – quaternion: %s, translation: %s", sensor_name,
                (qx, qy, qz, qw), (tx, ty, tz))
    return np.array([qx, qy, qz, qw], dtype=np.float64), np.array([tx, ty, tz], dtype=np.float64)

def load_intrinsics(parquet_path, clip_id, camera_name) :
    df = pd.read_parquet(parquet_path)
    row = df.loc[(clip_id, camera_name)]
    width, height, cx, cy = row[["width", "height", "cx", "cy"]].astype(float).values
    fw_coeffs = row[
        ["fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4"]
    ].astype(float).values
    fw_poly = fw_coeffs[::-1]
    logger.info(
        "Intrinsics for %s – size: %dx%d, principal point: (%.2f, %.2f)",
        camera_name,
        int(width),
        int(height),
        cx,
        cy,
    )
    return int(width), int(height), float(cx), float(cy), fw_poly


def build_dataset(cfg) :
    avdi = PhysicalAIAVDatasetInterface()
    sensor_name = cfg.camera_name 
    q, t = load_extrinsics(cfg.extrinsics_parquet, cfg.clip_id, sensor_name)
    R_wc = quat_to_rot(*q)  
    t_wc = t

    width, height, cx, cy, fw_poly = load_intrinsics(
        cfg.intrinsics_parquet, cfg.clip_id, sensor_name
    )

    egomotion = avdi.get_clip_feature(
        cfg.clip_id,
        avdi.features.LABELS.EGOMOTION,
        types="egomotion",
    )

    assert (
        cfg.t0_us
        > cfg.num_history_steps * cfg.time_step * 1_000_000
    ), "t0_us must be larger than the history time range"

    history_offsets_us = np.arange(
        -(cfg.num_history_steps - 1) * cfg.time_step * 1_000_000,
        cfg.time_step * 1_000_000 / 2,
        cfg.time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = cfg.t0_us + history_offsets_us

    future_offsets_us = np.arange(
        cfg.time_step * 1_000_000,
        (cfg.num_future_steps + 0.5) * cfg.time_step * 1_000_000,
        cfg.time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = cfg.t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation  
    ego_history_quat = ego_history.pose.rotation.as_quat()  

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation  
    ego_future_quat = ego_future.pose.rotation.as_quat() 

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)
    ).as_matrix()
    ego_future_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)
    ).as_matrix()

    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = (
        torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_rot_tensor = (
        torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)
    )

    camera_features = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    image_timestamps = np.array(
        [
            cfg.t0_us
            - (cfg.num_frames - 1 - i) * int(cfg.time_step * 1_000_000)
            for i in range(cfg.num_frames)
        ],
        dtype=np.int64,
    )

    image_frames_list: List[Tensor] = []
    camera_indices_list: List[int] = []
    timestamps_list: List[Tensor] = []

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(
            cfg.clip_id,
            cam_feature,
            types="camera",
        )
        frames_np, frame_ts = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = torch.from_numpy(frames_np)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        if isinstance(cam_feature, str):
            cam_name = cam_feature.split("/")[-1].lower()
        else:
            raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
        cam_idx = camera_name_to_index.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_ts.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6 

    data = {
        "image_frames": image_frames, 
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor, 
        "ego_history_rot": ego_history_rot_tensor, 
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor, 
        "relative_timestamps": relative_timestamps, 
        "absolute_timestamps": all_timestamps, 
        "t0_us": cfg.t0_us,
        "clip_id": cfg.clip_id,
        "intrinsics": {
            "width": width,
            "height": height,
            "cx": cx,
            "cy": cy,
            "fw_poly": fw_poly,
        },
        "extrinsics": {"R_wc": R_wc, "t_wc": t_wc},
    }

    logger.info("Dataset built successfully.")
    return data

def run_model(model, processor, data, device, dtype,) :
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    torch.cuda.manual_seed_all(42)
    with torch.autocast(device_type=device, dtype=dtype):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    return pred_xyz, pred_rot, extra

def draw_trajectory_on_image(img_path, output_path, pred_xyz, intrinsics, extrinsics, width, height,) :
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    R_wc = extrinsics["R_wc"]
    t_wc = extrinsics["t_wc"]
    fx_poly = intrinsics["fw_poly"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    pred_pts_world = pred_xyz.squeeze().cpu().numpy() 

    pixel_points: List[Tuple[int, int]] = []
    for pt_w in pred_pts_world:
        pt_c = world_to_cam(pt_w, R_wc, t_wc)
        proj = project_fisheye(pt_c, fx_poly, cx, cy)
        if proj is None:
            continue
        u, v = proj
        if 0 <= u < width and 0 <= v < height:
            pixel = (int(round(u)), int(round(v)))
            pixel_points.append(pixel)
            cv2.circle(img, pixel, radius=3, color=(0, 0, 255), thickness=-1) 

    if len(pixel_points) >= 2:
        cv2.polylines(
            img,
            [np.array(pixel_points, dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 0),
            thickness=2,
        )
    cv2.imwrite(str(output_path), img)
    logger.info("Trajectory visualisation saved to %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo‑R1 trajectory prediction."
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default="eed514a0-a366-4550-b9bd-4c296c531511",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/models/r1_10B"),
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("frontwide_with_trajectory.jpg"),
    )
    parser.add_argument(
        "--extracted-frame",
        type=Path,
        default=Path("frame_10s.jpg"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    cfg = Config(
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        model_dir=args.model_dir,
        output_image=args.output_image,
        extracted_frame=args.extracted_frame,
        verbose=args.verbose,
    )
    cfg.camera_name = "camera_front_wide_120fov"  

    video_path = (
        cfg.video_root
        / cfg.camera_name
        / f"{cfg.camera_name}.chunk_0000"
        / f"{cfg.clip_id}.{cfg.camera_name}.mp4"
    )
    extract_frame_cv2(
        video_path=video_path,
        timestamp_us=cfg.t0_us,
        output_path=cfg.extracted_frame,
        verbose=cfg.verbose,
    )

    data = build_dataset(cfg)

    logger.info("Loading Alpamayo‑R1 model from %s", cfg.model_dir)
    model = AlpamayoR1.from_pretrained(
        str(cfg.model_dir), dtype=cfg.dtype
    ).to(cfg.device)
    processor = helper.get_processor(model.tokenizer)
    pred_xyz, pred_rot, extra = run_model(
        model=model,
        processor=processor,
        data=data,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    logger.info("Inference completed.")
    logger.debug("Chain‑of‑thought: %s", extra.get("cot", ["<none>"])[0])
    
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    print("gt:",gt_xy)
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    print("pred_xy:",pred_xy)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    logger.info("minADE (using transpose): %.4f m", min_ade)

    draw_trajectory_on_image(
        img_path=cfg.extracted_frame,
        output_path=cfg.output_image,
        pred_xyz=pred_xyz,
        intrinsics=data["intrinsics"],
        extrinsics=data["extrinsics"],
        width=data["intrinsics"]["width"],
        height=data["intrinsics"]["height"],
    )

    logger.info("All done!")

if __name__ == "__main__":
    main()
