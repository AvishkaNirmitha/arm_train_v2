"""Camera indexes (/dev/videoN) and resolutions from the project's .env file, so a re-plug only means editing one file.

.env (project root, next to so101_real/), one KEY=value per line, # for comments:

    TOP_CAM=6
    WRIST_CAM=4
    BASE_CAM=2
    TOP_RES=1920x1080
    WRIST_RES=640x480
    BASE_RES=1920x1080
    CAM_FOURCC=MJPG

CAM_FOURCC: the pixel format asked from the cameras (MJPG = compressed, fast; YUYV = raw, ~1 fps at 1920x1080;
empty = the camera's choice). The resolution is the mode asked from the camera; top and base images are then centre-cropped to 4:3 (1920x1080 ->
1440x1080). The calibrations were measured in these modes: change one only together with a new calibration
(top: calibrate_top_camera.py, wrist: fit_wrist_points.py, base: calib_base_camera.py).

An environment variable of the same name wins over .env (e.g. `TOP_CAM=2 python ...`); a --top-cam / --wrist-cam /
--base-cam argument wins over both. Missing keys fall back to the defaults below.
"""
import os

ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
DEFAULTS = {"TOP_CAM": 2, "WRIST_CAM": 4, "BASE_CAM": None,      # BASE_CAM: no base camera unless .env names one
            "TOP_RES": "1920x1080", "WRIST_RES": "640x480", "BASE_RES": "1920x1080", "CAM_FOURCC": "MJPG"}


def _read_env_file(path=ENV_FILE):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def cam_index(key):
    """Camera index for key (TOP_CAM / WRIST_CAM / BASE_CAM): environment > .env > DEFAULTS. None = no such camera
    (BASE_CAM not set, or set to "none" / "off")."""
    val = os.environ.get(key, _read_env_file().get(key))
    if val is None:
        return DEFAULTS[key]
    if val.strip().lower() in ("", "none", "off"):
        return None
    try:
        return int(val)
    except ValueError:
        raise SystemExit(f"{ENV_FILE}: {key}={val!r} is not a camera index (a number, e.g. {key}=4)")


def cam_res(key):
    """(width, height) for key (TOP_RES / WRIST_RES / BASE_RES), written WIDTHxHEIGHT: environment > .env > DEFAULTS."""
    val = str(os.environ.get(key, _read_env_file().get(key, DEFAULTS[key])))
    try:
        w, h = (int(v) for v in val.lower().replace(" ", "").split("x"))
    except ValueError:
        raise SystemExit(f"{ENV_FILE}: {key}={val!r} is not a resolution (WIDTHxHEIGHT, e.g. {key}=1920x1080)")
    return w, h


TOP_CAM, WRIST_CAM, BASE_CAM = cam_index("TOP_CAM"), cam_index("WRIST_CAM"), cam_index("BASE_CAM")
TOP_RES, WRIST_RES, BASE_RES = cam_res("TOP_RES"), cam_res("WRIST_RES"), cam_res("BASE_RES")
CAM_FOURCC = str(os.environ.get("CAM_FOURCC", _read_env_file().get("CAM_FOURCC", DEFAULTS["CAM_FOURCC"]))).strip()
if CAM_FOURCC.lower() in ("", "none", "off"):
    CAM_FOURCC = ""
elif len(CAM_FOURCC) != 4:
    raise SystemExit(f"CAM_FOURCC={CAM_FOURCC!r}: 4 letters, e.g. MJPG or YUYV")
