# process_video — crop, matte, frames, semantic

IMAvatar `preprocess/preprocess.sh`에서 **DECA / landmarks / iris / FLAME optimize** 단계만 제외한 파이프라인.

## Output layout

```
{dataset_dir}/                          # e.g. .../flare_2/justin
  train.mp4
  test.mp4
  train_cropped.mp4                     # intermediate
  train_cropped_matte.mp4
  justin/                               # dataset_name == folder name
    train/
      image/1.png
      mask/1.png
      semantic/1.png                    # uint8 part id 0–18 (FLARE / train loader)
      semantic_color/1.png              # RGB visualization
    test/
      ...
```

`Config.input_dir` / `ImageDataset` expects **`{subject_root}/{scene}/image/*.png`** with sibling `mask/`, `semantic/`, `semantic_color/` — same as FLARE.

## Submodules (git submodules under `processing/process_video/`)

| Path | Repo | Weights |
|------|------|---------|
| `processing/process_video/MODNet/` | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) | [modnet_webcam_portrait_matting.ckpt](https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/view) → `MODNet/pretrained/` |
| `processing/process_video/face-parsing.PyTorch/` | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | [79999_iter.pth](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) → `face-parsing.PyTorch/res/cp/` |

```bash
git submodule update --init processing/process_video/MODNet processing/process_video/face-parsing.PyTorch
```

Layout:

```
processing/process_video/
  MODNet/
    pretrained/modnet_webcam_portrait_matting.ckpt
    demo/video_matting/custom/run.py   # may need IMAvatar patch
  face-parsing.PyTorch/
    res/cp/79999_iter.pth
    model.py
```

### MODNet note

`process_video.py` calls (from `MODNet` root):

```bash
python -m demo.video_matting.custom.run --video <cropped.mp4> --result-type matte --fps 25
```

Vanilla MODNet may not ship `demo.video_matting.custom.run`. If import fails, copy the **IMAvatar preprocess MODNet patch** from `reference_codes/IMAvatar/preprocess/submodules/MODNet` (or the project's FLARE preprocess tree) into this `submodules/MODNet`.

Expected matte output name: `{stem}_matte.mp4` next to `{stem}_cropped.mp4` (e.g. `train_cropped_matte.mp4`).

### System

```bash
sudo apt install ffmpeg
```

Python: `torch`, `torchvision`, `opencv-python`, `imageio`, `Pillow` (training env is fine).

## NeRFace / NeRFBlendShape → FLARE folders (no mp4)

Full preprocess: **copy PNG → MODNet image matte → face parsing**. No video / H.264 round-trip.

NeRFace:

```bash
python processing/process_video/process_nerface.py \
  --input-dir /Bean/data/gwangjin/2024/nbshapes/nerface \
  --output-dir /Bean/data/gwangjin/2024/nbshapes/flare_2
```

NeRFBlendShape (test = last 500 frames; overlap with train OK):

```bash
python processing/process_video/process_nerfblendshape.py \
  --input-dir /path/to/nerfblendshape_root \
  --output-dir /Bean/data/gwangjin/2024/nbshapes/flare_2 \
  --test-tail 500
```

Output layout (training ``--input-dir`` = inner subject folder):

```
flare_2/nf_01/nf_01/train/image/*.png
flare_2/nf_01/nf_01/train/mask/*.png
flare_2/nf_01/nf_01/train/semantic/*.png
flare_2/nbs_id1/nbs_id1/test/image/*.png
...
```

Resume: ``--skip-copy`` / ``--skip-matte`` / ``--skip-parse``.

Video-only subjects (e.g. FLARE ``justin/train.mp4``) still use ``process_video.py`` below.

## Usage

From repo root:

```bash
python processing/process_video/process_video.py \
  --dataset-dir /Bean/data/gwangjin/2024/nbshapes/flare_2/justin \
  --crop "1080:1080:420:0" \
  --resize 512 \
  --fps 25
```

Subset of videos:

```bash
python processing/process_video/process_video.py \
  --dataset-dir /path/to/justin \
  --videos train test
```

Resume after crop+matte already exist:

```bash
python processing/process_video/process_video.py \
  --dataset-dir /path/to/justin \
  --skip-crop --skip-matte
```

Face parsing only (frames already extracted):

```bash
python processing/process_video/parse_faces.py \
  --image-dir /path/to/justin/justin/train/image \
  --semantic-dir /path/to/justin/justin/train/semantic \
  --semantic-color-dir /path/to/justin/justin/train/semantic_color
```

## Crop / camera

`--crop` must match your capture resolution (IMAvatar example `1080:1080:420:0` is for a specific camera). Wrong crop → bad mattes and semantics.

Training camera (`assets/default_camera.npz`) is separate — run `processing/compute_camera_for_metrical_crop.py` after data prep.

## Semantic format

- **`semantic/`**: grayscale PNG, pixel value = part id (see `dataset/flare_semantic.py` / `dataset_util._load_semantic` comments).
- **`semantic_color/`**: overlay vis for debugging; optional for training (`paths_for_image` resolves it but losses use `semantic/`).

Part ids: 0 bg, 1 skin, 2/3 brows, 4/5 eyes, 6 glasses, 7/8 ears, 9 earring, 10 nose, 11 mouth, 12/13 lips, 14 neck, 15 necklace, 16 cloth, 17 hair, 18 hat.
