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

        os.makedirs("/data/predictions", exist_ok=True)

    def _inference(self, run_id: str) -> List[Tuple[Dict[str, Any], str]]:
        """Extracts data from line chart img.
        tuple[dict, str]:
            - dict: extracted data (keys are the individual data series)
            - str: path to img with the extracted data overlaying the img

        Predictions are saved under /data/{run_id}/predictions/lineformer
        """
        import cv2
        from lineformer import infer
        from lineformer import line_utils
        from pathlib import Path
        
        vol.reload()

        predictions = []

        os.makedirs(
            os.path.join(f"/data/{run_id}/predictions", "lineformer"),
            exist_ok=True,
        )

        print(f"files is in /data/{run_id}:", os.listdir(f"/data/{run_id}"))

        print(os.listdir(f"/data/{run_id}/predictions"))

        inputs = os.listdir(f"/data/{run_id}/input")

        img_paths = [os.path.join(Path(f"/data/{run_id}/input"), Path(img)) for img in inputs]

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
                    Path(f"/data/{run_id}/predictions/lineformer"),
                    Path(img_path.split("/")[-1]),
                )

                print("new img path", new_img_path)

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
