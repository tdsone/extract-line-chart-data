"""
Local ChartDete wrapper for detecting chart elements.
"""

import os
import json
import cv2
from pathlib import Path
from huggingface_hub import snapshot_download

from ..utils import logger


class ChartDete:
    def __init__(self, device: str = "cuda:0"):
        """Initialize ChartDete model."""
        from mmdet.apis import init_detector

        logger.info("Loading ChartDete model...")
        
        # Download model weights if not present
        model_dir = Path.home() / ".cache" / "plextract" / "chartdete"
        if not model_dir.exists():
            logger.info("Downloading ChartDete weights from HuggingFace...")
            snapshot_download("tdsone/chartdete", local_dir=str(model_dir))
        
        config_file = str(model_dir / "cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py")
        checkpoint_file = str(model_dir / "checkpoint.pth")
        
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.device = device
        logger.info("Successfully loaded ChartDete!")

    def inference(self, input_dir: str, output_dir: str) -> None:
        """
        Detect chart elements in all images in input_dir.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
        """
        from mmdet.apis import inference_detector

        logger.info("Running ChartDete...")

        inputs = os.listdir(input_dir)
        img_paths = [os.path.join(input_dir, img) for img in inputs]
        results_base_folders = [Path(output_dir) / img / "chartdete" for img in inputs]

        for img_path, results_base_folder in zip(img_paths, results_base_folders):
            try:
                predictions = inference_detector(self.model, img_path)

                result_path = str(results_base_folder / "predictions.jpg")

                self.model.show_result(
                    img_path,
                    predictions,
                    out_file=result_path,
                )

                results_labelled = {}
                labels = [
                    "x_title",
                    "y_title",
                    "plot_area",
                    "other",
                    "xlabel",
                    "ylabel",
                    "chart_title",
                    "x_tick",
                    "y_tick",
                    "legend_patch",
                    "legend_label",
                    "legend_title",
                    "legend_area",
                    "mark_label",
                    "value_label",
                    "y_axis_area",
                    "x_axis_area",
                    "tick_grouping",
                ]
                for res, label in zip(predictions, labels):
                    results_labelled[label] = res.tolist()

                # Save coordinates to json
                with open(results_base_folder / "bounding_boxes.json", "w") as f:
                    json.dump(results_labelled, f)

                # Load image for cropping
                image = cv2.imread(img_path)

                if image is None:
                    logger.error(f"Error loading image: {img_path}")
                    continue

                bounding_boxes = results_labelled
                confidence_threshold = 0.9
                label_coordinates = {}

                plot_areas = sorted(
                    bounding_boxes["plot_area"], key=lambda el: el[4], reverse=True
                )

                if plot_areas:
                    highest_conf_pa = plot_areas[0]
                    label_coordinates["plot_area"] = highest_conf_pa

                # Function to crop bounding boxes
                def crop_bounding_boxes(image, boxes, threshold):
                    cropped_images = []
                    for box in boxes:
                        x1, y1, x2, y2, confidence = box
                        if confidence >= threshold:
                            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                            cropped_images.append(cropped_image)
                    return cropped_images

                # Crop bounding boxes from both 'xlabel' and 'ylabel'
                cropped_x_labels = crop_bounding_boxes(
                    image, bounding_boxes["xlabel"], confidence_threshold
                )
                cropped_y_labels = crop_bounding_boxes(
                    image, bounding_boxes["ylabel"], confidence_threshold
                )

                # Save cropped images
                for i, cropped_image in enumerate(cropped_x_labels):
                    path = str(results_base_folder / f"cropped_xlabels_{i}.jpg")
                    cv2.imwrite(path, cropped_image)
                    label_coordinates[path] = bounding_boxes["xlabel"][i]

                for i, cropped_image in enumerate(cropped_y_labels):
                    path = str(results_base_folder / f"cropped_ylabels_{i}.jpg")
                    cv2.imwrite(path, cropped_image)
                    label_coordinates[path] = bounding_boxes["ylabel"][i]

                with open(results_base_folder / "label_coordinates.json", "w") as f:
                    json.dump(label_coordinates, f)

                logger.info(f"ChartDete: Processed {Path(img_path).name}")

            except Exception as e:
                logger.error(f"ChartDete failed for {img_path}: {e}")
                raise

