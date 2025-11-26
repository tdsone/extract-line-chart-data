import os
import subprocess
from modal import App, Volume, Image

APP_NAME = "plextract" 
VOLUME_NAME = "plextract-vol"

modal_app = App(APP_NAME)
vol = Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_cv_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .run_commands("git clone https://github.com/tdsone/LineFormer.git")
    .pip_install(
        "openmim",
        "chardet",
        "transformers~=4.38.2",
        "bresenham",
        "tqdm",
        "torch==1.13.1",
        "torchvision==0.14.1",
    )
    .run_commands("mim install mmcv-full")
    .pip_install(
        "scikit-image",
        "matplotlib",
        "opencv-python",
        "pillow",
        "scipy==1.9.3",
    )
    .run_commands("pip install -e LineFormer/mmdetection")
)


def download_volume_dir(
    remote_dir: str,
    local_dir: str,
    volume_name: str = VOLUME_NAME,
) -> None:
    """
    Download all files under `remote_dir` inside the Modal Volume `volume_name`
    into the local directory `local_dir`, preserving subdirectory structure.

    - volume_name: name of the Modal Volume (as shown in `modal volume list`)
    - remote_dir: path inside the volume, e.g. "checkpoints/run-1" or "/"
    - local_dir: local destination directory
    """
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Normalise remote_dir
    remote_dir = remote_dir.rstrip("/")
    if remote_dir == "":
        remote_dir = "/"
    
    # Use Modal CLI to download the directory
    # Format: modal volume get <volume-name> <remote-path> <local-path>
    
    result = subprocess.run(
        ["modal", "volume", "get", volume_name, remote_dir, local_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download volume directory: {result.stderr}\n"
            f"Command: modal volume get {volume_name} {remote_dir} {local_dir}"
        )