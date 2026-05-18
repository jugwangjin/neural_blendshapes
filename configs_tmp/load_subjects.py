"""Parse configs_tmp/*.txt into subject entries (input_dir, train_dir, eval_dir)."""

import ast
from pathlib import Path

CONFIGS_TMP_DIR = Path(__file__).resolve().parent

# Server mount aliases (first existing wins in resolve_data_path)
BEAN_DATA_ROOTS = [
    Path("/Bean/data"),
    Path("//bean.postech.ac.kr/data"),
    Path(r"\\bean.postech.ac.kr\data"),
]

DEFAULT_FLARE2_ROOT = Path(r"\\bean.postech.ac.kr\data\gwangjin\2024\nbshapes\flare_2")


def resolve_data_path(path_str):
    p = Path(path_str)
    if p.exists():
        return p

    s = path_str.replace("\\", "/")
    if "/Bean/data/" in s:
        suffix = s.split("/Bean/data/", 1)[1]
        for root in BEAN_DATA_ROOTS:
            candidate = root / suffix
            if candidate.exists():
                return candidate
    return p


def _parse_config_line(line):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None, None
    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip()
    if val.startswith("["):
        val = ast.literal_eval(val)
    elif val.replace(".", "", 1).isdigit() or (
        val.startswith("-") and val[1:].replace(".", "", 1).isdigit()
    ):
        val = float(val) if "." in val else int(val)
    return key, val


def load_subject_config(config_path):
    data = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            key, val = _parse_config_line(line)
            if key:
                data[key] = val
    return data


def load_all_subjects(configs_dir=None):
    configs_dir = Path(configs_dir or CONFIGS_TMP_DIR)
    subjects = {}
    for path in sorted(configs_dir.glob("*.txt")):
        cfg = load_subject_config(path)
        name = cfg.get("run_name", path.stem)
        input_dir = cfg.get("input_dir")
        if not input_dir:
            continue
        train_dir = cfg.get("train_dir", [])
        eval_dir = cfg.get("eval_dir", [])
        if isinstance(train_dir, str):
            train_dir = [train_dir]
        if isinstance(eval_dir, str):
            eval_dir = [eval_dir]
        scenes = list(train_dir) + list(eval_dir)
        subjects[name] = {
            "config_path": path,
            "input_dir": resolve_data_path(str(input_dir)),
            "input_dir_raw": str(input_dir),
            "train_dir": train_dir,
            "eval_dir": eval_dir,
            "scenes": scenes,
        }
    return subjects


def list_flare2_subject_dirs(flare2_root=None):
    root = Path(flare2_root or DEFAULT_FLARE2_ROOT)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
