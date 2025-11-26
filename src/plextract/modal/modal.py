from modal import App, Volume, Image

app = App("plextract")

vol = Volume.from_name("plextract-vol", create_if_missing=True)


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
