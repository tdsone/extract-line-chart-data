"""
Local LineFormer wrapper for extracting line data from charts.
"""

import os
import json
import cv2
from pathlib import Path
from huggingface_hub import snapshot_download

from ..utils import logger


class LineFormer:
    def __init__(self, device: str = "cuda:0"):
        """Initialize LineFormer model."""
        from lineformer import infer

        logger.info("Loading LineFormer model...")
        
        # Download model weights if not present
        model_dir = Path.home() / ".cache" / "plextract" / "lineformer"
        if not model_dir.exists():
            logger.info("Downloading LineFormer weights from HuggingFace...")
            snapshot_download("tdsone/lineformer", local_dir=str(model_dir))
        
        ckpt = str(model_dir / "iter_3000.pth")
        config = str(model_dir / "lineformer_swin_t_config.py")
        
        infer.load_model(config, ckpt, device)
        logger.info("Successfully loaded LineFormer!")

    def inference(self, input_path: str, output_dir: str) -> None:
        """
        Extract line data from a chart image.
        
        Args:
            input_path: Path to the input image
            output_dir: Directory to save results (will create lineformer/ subdirectory)
        """
        from lineformer import infer
        from lineformer import line_utils

        img_name = Path(input_path).name
        results_base_folder = Path(output_dir) / img_name / "lineformer"
        
        try:
            os.makedirs(results_base_folder, exist_ok=True)

            img = cv2.imread(input_path)  # BGR format
            line_dataseries = infer.get_dataseries(img, to_clean=False)

            # Visualize extracted line keypoints
            img_viz = line_utils.draw_lines(
                img, line_utils.points_to_array(line_dataseries)
            )

            cv2.imwrite(str(results_base_folder / "prediction.png"), img_viz)

            with open(results_base_folder / "coordinates.json", "w") as f:
                json.dump(line_dataseries, f)

            logger.info(f"LineFormer: Processed {img_name}")

        except Exception as e:
            logger.error(f"Failed to make prediction for {input_path}: {e}")
            raise

