import os
from plextract.modal import vol, base_cv_image
from typing import List, Dict, Tuple, Any

from modal import Image, App, build, enter, gpu, method, Mount


lineformer_image = base_cv_image.run_commands("pip install -e LineFormer")

app = App("plextract-lineformer")


@app.cls(
    gpu=gpu.A10G(),
    container_idle_timeout=240,
    image=lineformer_image,
    volumes={"/data": vol},
)
class LineFormer:
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

    def _inference(self, run_id: str, img: str) -> None:
        """Extracts data from line chart img.

        Predictions are saved under /data/{run_id}/ouptut/{img}/lineformer
        """
        import cv2
        from lineformer import infer
        from lineformer import line_utils
        from pathlib import Path
        import json

        vol.reload()

        BASE_INPUT = f"/data/{run_id}/input"
        BASE_OUTPUT = f"/data/{run_id}/output"

        try:
            os.makedirs(
                f"{BASE_OUTPUT}/{img}/lineformer",
                exist_ok=True,
            )

            img_path = f"{BASE_INPUT}/{img}"
            results_base_folder = f"{BASE_OUTPUT}/{img}/lineformer"

            img = cv2.imread(img_path)  # BGR format

            line_dataseries = infer.get_dataseries(img, to_clean=False)

            # Visualize extracted line keypoints
            img = line_utils.draw_lines(
                img, line_utils.points_to_array(line_dataseries)
            )

            cv2.imwrite(f"{results_base_folder}/prediction.png", img)

            with open(f"{results_base_folder}/coordinates.json", "w") as f:
                json.dump(line_dataseries, f)

            vol.commit()

        except Exception as e:
            print(f"Failed to make prediction for {img_path}.")
            print(e)

    @method()
    def inference(self, run_id: str, img: str):
        return self._inference(run_id=run_id, img=img)
