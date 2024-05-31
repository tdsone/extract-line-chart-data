import os
from plextract.modal import app, vol
from typing import List, Dict, Tuple, Any

from modal import Image, build, enter, gpu, method, Mount


lineformer_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .run_commands("git clone https://github.com/tdsone/LineFormer.git")
    .pip_install(
        "openmim",
        "chardet",
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
    .pip_install(
        "bresenham",
        "tqdm",
        "transformers~=4.38.2",
    )
    .run_commands("pip install -e LineFormer")
)


@app.cls(
    gpu=gpu.A10G(),
    container_idle_timeout=240,
    image=lineformer_image,
    volumes={"/predictions": vol},
    mounts=[Mount.from_local_dir("input", remote_path="/input")],
)
class Model:
    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        import os

        os.makedirs("huggingface")

        snapshot_download("tdsone/lineformer", local_dir="huggingface")

    @enter()
    def enter(self):
        from lineformer import infer

        CKPT = "/root/huggingface/iter_3000.pth"
        CONFIG = "/root/huggingface/lineformer_swin_t_config.py"
        DEVICE = "cuda:0"

        infer.load_model(CONFIG, CKPT, DEVICE)

        os.makedirs("predictions", exist_ok=True)

    def _inference(self, run_id: str) -> List[Tuple[Dict[str, Any], str]]:
        """Extracts data from line chart img.
        tuple[dict, str]:
            - dict: extracted data (keys are the individual data series)
            - str: path to img with the extracted data overlaying the img

        Predictions are saved under predictions/{run_id}
        """
        import cv2
        from lineformer import infer
        from lineformer import line_utils
        from pathlib import Path

        predictions = []

        os.makedirs(os.path.join(Path("/predictions"), Path(run_id)), exist_ok=False)

        inputs = os.listdir("/input")

        img_paths = [os.path.join(Path("/input"), Path(img)) for img in inputs]

        for img_path in img_paths:
            try:

                img = cv2.imread(img_path)  # BGR format

                line_dataseries = infer.get_dataseries(img, to_clean=False)

                # Visualize extracted line keypoints
                img = line_utils.draw_lines(
                    img, line_utils.points_to_array(line_dataseries)
                )

                # Construct the new path to save the image
                new_img_path = os.path.join(
                    Path(f"/predictions/{run_id}"), Path(img_path.split("/")[2])
                )

                cv2.imwrite(new_img_path, img)
                vol.commit()

                predictions.append((line_dataseries, new_img_path))
            except Exception as e:
                print(f"Failed to make prediction for {img_path}.")
                print(e)

        return predictions

    @method()
    def inference(self, run_id: str):
        return self._inference(run_id=run_id)
