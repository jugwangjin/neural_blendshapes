"""MediaPipe index → ICT projection component routing."""

from processing.ict_mediapipe_lmk.constants import LEFT_IRIS_MP, RIGHT_IRIS_MP

# Eye contour / eyelid (478 mesh). Excludes iris 468–477.
LEFT_EYELID_MP = frozenset(
    [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30, 247, 226, 35,
    ]
)
RIGHT_EYELID_MP = frozenset(
    [
        362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
        341, 256, 441, 442, 443, 444, 445, 342, 446, 414, 286, 258, 257, 259, 260, 467, 359, 255,
    ]
)

LEFT_IRIS_MP_SET = frozenset(LEFT_IRIS_MP)
RIGHT_IRIS_MP_SET = frozenset(RIGHT_IRIS_MP)
IRIS_MP_SET = frozenset(range(468, 478))

# geometry_chart_id for 3DGS / texture chart separation
CHART_FACE = 0
CHART_LEFT_EYE = 1
CHART_RIGHT_EYE = 2


def classify_mp_landmark(mp_idx: int) -> str:
    mp_idx = int(mp_idx)
    if mp_idx in LEFT_IRIS_MP_SET:
        return "left_iris"
    if mp_idx in RIGHT_IRIS_MP_SET:
        return "right_iris"
    if mp_idx in LEFT_EYELID_MP:
        return "left_eyelid"
    if mp_idx in RIGHT_EYELID_MP:
        return "right_eyelid"
    return "face"


def geometry_chart_id(target_type: str) -> int:
    if target_type in ("left_iris", "left_eyelid"):
        return CHART_LEFT_EYE
    if target_type in ("right_iris", "right_eyelid"):
        return CHART_RIGHT_EYE
    return CHART_FACE
