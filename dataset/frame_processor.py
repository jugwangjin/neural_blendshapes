"""Run MediaPipe + face_alignment once per frame; cache ``.npz``; skip no-face frames."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset.dataset_util import matrix_to_pose_feat, parse_mediapipe_output, paths_for_image


def _load_rgb_uint8(img_path: Path):
    import cv2

    bgr = cv2.imread(str(img_path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FrameProcessor:
    def __init__(self, face_landmarker_task: Path, fa_scale: float = 1.25):
        import mediapipe as mp

        self.fa_scale = fa_scale
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_landmarker_task)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.mediapipe = FaceLandmarker.create_from_options(options)

        import face_alignment

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_alignment = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D,
            flip_input=False,
            device=device,
        )

    def process_rgb(self, rgb: np.ndarray):
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.mediapipe.detect(mp_image)
        mp_lm, mp_bs, mp_T = parse_mediapipe_output(result)
        if mp_lm is None:
            return None

        landmarks, scores, _ = self.face_alignment.get_landmarks_from_image(
            rgb,
            return_bboxes=True,
            return_landmark_score=True,
        )
        if landmarks is None or len(landmarks) == 0:
            return None

        fa_lm = torch.tensor(landmarks[0], dtype=torch.float32)
        fa_score = torch.tensor(scores[0], dtype=torch.float32)
        if fa_lm.shape[-1] >= 3:
            fa_lm = torch.cat([fa_lm[..., :3], fa_score.unsqueeze(-1)], dim=-1)
        else:
            fa_lm = torch.cat([fa_lm[..., :2], fa_score.unsqueeze(-1)], dim=-1)

        h, w = rgb.shape[:2]
        mp_pose_raw = matrix_to_pose_feat(mp_T)[:6]
        pose_feat = matrix_to_pose_feat(mp_T).numpy()

        return {
            "mp_blendshape": mp_bs.numpy().astype(np.float32),
            "mp_landmarks_2d": mp_lm[:, :2].numpy().astype(np.float32),
            "mp_landmarks_3d": mp_lm.numpy().astype(np.float32),
            "mp_transform_matrix": mp_T.numpy().astype(np.float32),
            "mp_pose_raw": mp_pose_raw.numpy().astype(np.float32),
            "pose_feat": pose_feat.astype(np.float32),
            "mp_valid": np.ones(478, dtype=np.float32),
            "landmark_fa": fa_lm.numpy().astype(np.float32),
            "image_hw": np.array([h, w], dtype=np.int32),
        }

    def process_image_file(self, img_path: Path):
        rgb = _load_rgb_uint8(img_path)
        return self.process_rgb(rgb)


def cache_path_for_image(cfg, split: str, img_path: Path) -> Path:
    subject = Path(cfg.input_dir).name
    return Path(cfg.mp_cache_dir) / subject / split / f"{img_path.stem}.npz"


def build_split_cache(
    cfg,
    split: str,
    image_paths,
    *,
    rebuild: bool = False,
    face_landmarker_task: Path = None,
):
    """
    MediaPipe + face_alignment for every image; only valid faces indexed.
    Returns (valid_image_paths, cache_npz_paths).
    """
    task = Path(face_landmarker_task or cfg.face_landmarker_task)
    processor = FrameProcessor(task, fa_scale=getattr(cfg, "fa_bbox_scale", 1.25))

    valid_images = []
    valid_caches = []
    skipped = 0

    for img_path in tqdm(image_paths, desc=f"MP+FA cache [{split}]"):
        img_path = Path(img_path)
        out_npz = cache_path_for_image(cfg, split, img_path)
        out_npz.parent.mkdir(parents=True, exist_ok=True)

        if out_npz.is_file() and not rebuild:
            data = np.load(out_npz, allow_pickle=True)
            if bool(data.get("valid", True)):
                valid_images.append(img_path)
                valid_caches.append(out_npz)
            else:
                skipped += 1
            continue

        payload = processor.process_image_file(img_path)
        if payload is None:
            np.savez(out_npz, valid=np.bool_(False))
            skipped += 1
            continue

        np.savez(
            out_npz,
            valid=np.bool_(True),
            **{k: v for k, v in payload.items() if k != "image_hw"},
        )
        valid_images.append(img_path)
        valid_caches.append(out_npz)

    print(f"[{split}] valid={len(valid_images)} skipped(no face)={skipped}")
    return valid_images, valid_caches
