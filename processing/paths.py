"""Shared repo / processing paths and sys.path bootstrap."""

import os
import sys
from pathlib import Path

PROCESSING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROCESSING_ROOT.parent

FLAME_MODEL = PROCESSING_ROOT / "flame" / "FLAME2020" / "generic_model.pkl"
FLAME_STATIC_EMBEDDING = REPO_ROOT / "assets" / "flame_static_embedding.pkl"
FLAME_UV_MESH = REPO_ROOT / "assets" / "canonical_eye_smpl.obj"
ICT_NPY = REPO_ROOT / "assets" / "ict_facekit_torch.npy"
ICT_CANONICAL = REPO_ROOT / "assets" / "ict_identity.npy"
ASSETS_DIR = REPO_ROOT / "assets"

METRICAL_ROOT = PROCESSING_ROOT / "metrical-tracker"
LARGE_STEPS_ROOT = PROCESSING_ROOT / "large-steps-pytorch"

ICT_FACEKIT_PKG = None
ICT_FACEX_MODEL = None


def _has_facekit_scripts(pkg_dir: Path) -> bool:
    return (pkg_dir / "Scripts" / "face_model_io.py").is_file()


def resolve_ict_facekit():
    """
    Locate ICT_FaceKit Python package (gitignored; clone separately).

    Returns (repo_root_on_syspath, pkg_dir, facex_model_dir).
    repo_root_on_syspath is the directory that must be on sys.path so that
    ``from ICT_FaceKit.Scripts import face_model_io`` works.
    """
    env_root = os.environ.get("ICT_FACEKIT_ROOT", "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            REPO_ROOT,
            REPO_ROOT / "ICT-FaceKit",
            REPO_ROOT.parent / "ICT_FaceKit",
            REPO_ROOT.parent / "ICT-FaceKit",
        ]
    )

    for repo_root in candidates:
        if not repo_root:
            continue
        pkg = repo_root / "ICT_FaceKit"
        if _has_facekit_scripts(pkg):
            facex = pkg / "FaceXModel"
            return repo_root.resolve(), pkg.resolve(), facex.resolve()

    default_pkg = REPO_ROOT / "ICT_FaceKit"
    return REPO_ROOT.resolve(), default_pkg.resolve(), (default_pkg / "FaceXModel").resolve()


def _prepend_syspath(path: Path):
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def setup_ict_facekit_import():
    """
    Configure sys.path for USC ICT-FaceKit.

    - REPO_ROOT (or ICT_FACEKIT_ROOT): ``from ICT_FaceKit.Scripts import face_model_io``
    - ICT_FaceKit/Scripts: ``import ict_face_model`` inside face_model_io.py
    """
    global ICT_FACEKIT_PKG, ICT_FACEX_MODEL
    repo_root, pkg, facex = resolve_ict_facekit()
    scripts_dir = pkg / "Scripts"
    if not _has_facekit_scripts(pkg):
        raise FileNotFoundError(
            "ICT_FaceKit not found.\n"
            f"  Expected package at: {REPO_ROOT / 'ICT_FaceKit'}/Scripts/face_model_io.py\n"
            "  Clone the official ICT-FaceKit repo into the project root as ICT_FaceKit, e.g.:\n"
            f"    cd {REPO_ROOT}\n"
            "    git clone https://github.com/USC-ICT/ICT-FaceKit.git ICT_FaceKit\n"
            "  Or set ICT_FACEKIT_ROOT to the parent directory that contains the ICT_FaceKit/ folder."
        )
    _prepend_syspath(repo_root)
    _prepend_syspath(scripts_dir)
    ICT_FACEKIT_PKG = pkg
    ICT_FACEX_MODEL = facex
    return str(pkg), str(facex)


def setup_import_paths():
    for root in (REPO_ROOT, PROCESSING_ROOT):
        path = str(root)
        if path not in sys.path:
            sys.path.insert(0, path)
