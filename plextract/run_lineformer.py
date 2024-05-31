import os
from plextract.modal import app, vol
from typing import List, Dict, Tuple, Any

from modal import (
    Image,
    build,
    enter,
    gpu,
    method,
)


lineformer_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "openmim",
        "chardet",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "scikit-image",
        "matplotlib",
        "opencv-python",
        "pillow",
        "scipy==1.9.3",
        "bresenham",
        "tqdm",
        "transformers~=4.38.2",
    )
    .run_commands("mim install mmcv-full")
    .run_commands(
        "git clone https://github.com/tdsone/LineFormer.git", force_build=True
    )
    .run_commands("pip install -e LineFormer/mmdetection")
    .run_commands("pip install -e LineFormer")
)

with lineformer_image.imports():
    import cv2


@app.cls(
    gpu=gpu.A10G(),
    container_idle_timeout=240,
    image=lineformer_image,
    volumes={"/predictions": vol},
)
class Model:
    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        snapshot_download("tdsone/lineformer")

    @enter()
    def enter(self):
        from lineformer import infer

        CKPT = "iter_3000.pth"
        CONFIG = "lineformer_swin_t_config.py"
        DEVICE = "cuda:0"

        infer.load_model(CONFIG, CKPT, DEVICE)

        os.makedirs("predictions", exist_ok=True)

        print(os.listdir())

    def _inference(
        self, run_id: str, img_paths: List[str]
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Extracts data from line chart img.
        tuple[dict, str]:
            - dict: extracted data (keys are the individual data series)
            - str: path to img with the extracted data overlaying the img

        Predictions are saved under predictions/{run_id}
        """
        from lineformer import infer
        from lineformer import line_utils

        predictions = []

        os.makedirs(run_id, exist_ok=False)

        for img_path in img_paths:
            try:
                img = cv2.imread(img_path)  # BGR format

                line_dataseries = infer.get_dataseries(img, to_clean=False)

                # Visualize extracted line keypoints
                img = line_utils.draw_lines(
                    img, line_utils.points_to_array(line_dataseries)
                )

                # Construct the new path to save the image
                relative_path = os.path.relpath(img_path, run_id)
                new_img_path = os.path.join("predictions", relative_path)

                cv2.imwrite(new_img_path, img)

                predictions.append(line_dataseries, new_img_path)
            except Exception as e:
                print(f"Failed to make prediction for {img_path}.")
                print(e)

        return predictions

    @method()
    def inference(self, img_path: List[str]):
        return self._inference(img_path=img_path)
