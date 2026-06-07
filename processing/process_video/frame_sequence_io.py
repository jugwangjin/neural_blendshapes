"""Sort frame files by numeric stem and encode image sequences to mp4 (ffmpeg)."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.frame_sort import frame_sort_key, sort_frame_paths


def stem_int(path: Path) -> int:
    key = frame_sort_key(path)
    if key[0] != 0:
        raise ValueError(f"expected numeric stem, got {path.name}")
    return int(key[1])


def sort_frames_numeric(paths: list[Path]) -> list[Path]:
    return sort_frame_paths(paths)


def collect_images(
    image_dir: Path,
    *,
    suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".JPG", ".JPEG"),
) -> list[Path]:
    out = []
    for suf in suffixes:
        out.extend(image_dir.glob(f"*{suf}"))
    return sort_frames_numeric(out)


def collect_png_frames(image_dir: Path) -> list[Path]:
    """Numeric-sort ``*.png`` under ``image_dir`` (tracking / eval frame dumps)."""
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if len(paths) == 0:
        return []
    return sort_frames_numeric(paths)


_ENCODER_CACHE: str | None = None

_ENCODER_ORDER = ("libx264", "h264_nvenc", "mpeg4", "libopenh264")


def _ffmpeg_encoders_text() -> str:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _available_video_encoders() -> set[str]:
    enc = _ffmpeg_encoders_text()
    return {name for name in _ENCODER_ORDER if re.search(rf"\s+{re.escape(name)}\s+", enc)}


def _encoder_candidates(preferred: str | None = None) -> list[str]:
    if preferred is not None and preferred.strip():
        return [preferred.strip()]
    avail = _available_video_encoders()
    return [name for name in _ENCODER_ORDER if name in avail]


def resolve_video_encoder(preferred: str | None = None) -> str:
    """
    Pick an H.264-ish encoder available in the active ``ffmpeg``.

    Conda builds often ship ``libopenh264`` but not ``libx264``; the bundled
    OpenH264 .so frequently mismatches at runtime — ``mpeg4`` is tried first
    among always-built-in fallbacks (see ``_ENCODER_ORDER``).
    """
    global _ENCODER_CACHE
    if preferred is not None and preferred.strip():
        return preferred.strip()
    if _ENCODER_CACHE is not None:
        return _ENCODER_CACHE
    candidates = _encoder_candidates()
    if not candidates:
        raise RuntimeError(
            "no supported ffmpeg video encoder (tried libx264, h264_nvenc, mpeg4, libopenh264)"
        )
    _ENCODER_CACHE = candidates[0]
    print(f"[frame_sequence_io] ffmpeg encoder: {_ENCODER_CACHE}", flush=True)
    return _ENCODER_CACHE


def _rgba_flatten_vf(background: tuple[int, int, int]) -> str:
    """Straight-alpha RGBA composited onto ``background`` (default use: white mp4 backdrop)."""
    br, bg, bb = background
    a = "alpha(X,Y)"

    def _ch(name: str, back: int) -> str:
        return f"({name}(X,Y)*{a}/255+{back}*(255-{a})/255)"

    return (
        "format=rgba,"
        f"geq=r='{_ch('r', br)}':g='{_ch('g', bg)}':b='{_ch('b', bb)}',"
        "format=yuv420p"
    )


def _ffmpeg_encode_cmd(
    *,
    list_path: Path,
    out_mp4: Path,
    codec: str,
    fps: int,
    flatten_rgba: bool = True,
    video_background: tuple[int, int, int] = (255, 255, 255),
) -> list[str]:
    if flatten_rgba:
        vf = _rgba_flatten_vf(video_background)
    else:
        vf = "format=yuv420p"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        vf,
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        cmd.extend(["-preset", "fast"])
    elif codec in ("libopenh264", "mpeg4"):
        cmd.extend(["-b:v", "4M"])
    cmd.extend(["-pix_fmt", "yuv420p", "-r", str(int(fps)), str(out_mp4)])
    return cmd


def images_to_mp4(
    image_paths: list[Path],
    out_mp4: Path,
    *,
    fps: int = 25,
    video_codec: str | None = None,
    video_background: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Encode PNG/JPEG frame list to mp4.

    Frame PNGs may be straight-alpha RGBA; mp4 (yuv420p) composites them onto
    ``video_background`` (default white). Opaque RGB frames pass through unchanged.
    """
    if len(image_paths) == 0:
        raise ValueError(f"no frames to encode -> {out_mp4}")
    image_paths = sort_frame_paths([Path(p) for p in image_paths])
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / float(fps)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:
        list_path = Path(f.name)
        for p in image_paths:
            line = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{line}'\n")
            f.write(f"duration {duration}\n")
        last = str(image_paths[-1].resolve()).replace("'", "'\\''")
        f.write(f"file '{last}'\n")

    global _ENCODER_CACHE
    candidates = _encoder_candidates(video_codec)
    if not candidates:
        list_path.unlink(missing_ok=True)
        raise RuntimeError(
            "no supported ffmpeg video encoder (tried libx264, h264_nvenc, mpeg4, libopenh264)"
        )

    last_stderr = ""
    for codec in candidates:
        for flatten_rgba in (True, False):
            cmd = _ffmpeg_encode_cmd(
                list_path=list_path,
                out_mp4=out_mp4,
                codec=codec,
                fps=fps,
                flatten_rgba=flatten_rgba,
                video_background=video_background,
            )
            print("+", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                _ENCODER_CACHE = codec
                print(f"[frame_sequence_io] encoded with {codec}", flush=True)
                list_path.unlink(missing_ok=True)
                return
            last_stderr = (proc.stderr or "") + (proc.stdout or "")
            if flatten_rgba and "No such filter: 'geq'" in last_stderr:
                continue
            break
        if codec != candidates[-1]:
            tail = last_stderr.strip().splitlines()
            hint = tail[-1] if tail else "unknown error"
            print(f"[frame_sequence_io] {codec} failed ({hint}); trying next encoder...", flush=True)
            if _ENCODER_CACHE == codec:
                _ENCODER_CACHE = None

    list_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"ffmpeg encode failed (tried {candidates}): {last_stderr[-3000:]}"
    )
