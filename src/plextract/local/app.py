"""
Local pipeline orchestration for chart data extraction.
"""

import os
import json
from pathlib import Path

from ..utils import logger, correct_coordinates
from .lineformer import LineFormer
from .chartdete import ChartDete
from .trocr import OCRModel


def run_pipeline(input_dir: str, output_dir: str) -> None:
    """
    Run the full chart data extraction pipeline locally.
    
    Args:
        input_dir: Directory containing input chart images
        output_dir: Directory to save all outputs
    """
    logger.info(f"Running local pipeline...")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    input_files = os.listdir(input_dir)
    logger.debug(f"Found input files: {input_files}")

    # Create output folders for each input file
    logger.info("Creating output folders...")
    for file in input_files:
        file_dir = Path(output_dir) / file
        os.makedirs(file_dir / "chartdete", exist_ok=True)
        os.makedirs(file_dir / "lineformer", exist_ok=True)

    # Step 1: Extract lines from images using LineFormer
    logger.info("Extracting lines from images using LineFormer...")
    lineformer = LineFormer()
    for img in input_files:
        img_path = os.path.join(input_dir, img)
        lineformer.inference(img_path, output_dir)

    # Step 2: Detect chart elements using ChartDete
    logger.info("Detecting chart elements using ChartDete...")
    chartdete = ChartDete()
    chartdete.inference(input_dir, output_dir)

    # Step 3: OCR text from axis labels
    logger.info("Running OCR on axis labels...")
    axis_label_images = []
    for plot_img_dir in os.listdir(output_dir):
        chartdete_dir = Path(output_dir) / plot_img_dir / "chartdete"
        if not chartdete_dir.exists():
            continue

        for label_img in os.listdir(chartdete_dir):
            if "label" in label_img and ".json" not in label_img:
                axis_label_images.append(str(chartdete_dir / label_img))

    ocr_model = OCRModel()
    label_texts = ocr_model.inference_batch(axis_label_images)

    # Save OCR results to file for each image
    for img_dir in os.listdir(output_dir):
        path = Path(output_dir) / img_dir / "axis_label_texts.json"
        logger.info(f"Saving OCR results to {path}...")
        with open(path, "w") as f:
            json.dump(
                {
                    img_path: extracted_text
                    for img_path, extracted_text in label_texts
                    if img_dir in img_path
                },
                f,
            )

    # Step 4: Correct coordinates (convert pixel coords to actual values)
    logger.info("Correcting coordinates...")
    for img in input_files:
        correct_coordinates(output_dir, img)

    logger.info("Local pipeline complete!")

