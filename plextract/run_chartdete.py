"""
This script extracts data from plots using LineFormer and ChartDete.

Setup:
- If you have not used modal before, you can set up an account here: modal.com
- Currently the weights are not downloaded using this code.

Run this script using: modal run chartdete/main.py

If you struggle to set this up, don't hesitate to shoot me an email: mail@timonschneider.de
"""

from pathlib import Path
from modal import enter, build, Mount, App, method

from plextract.modal import base_cv_image, vol

chardete_image = base_cv_image.run_commands(
    "git clone https://github.com/tdsone/ChartDete"
).run_commands("pip install -e ChartDete")

app = App("plextract-chartdete")


@app.cls(
    image=chardete_image,
    timeout=3000,
    gpu="any",
    volumes={"/data": vol},
)
class ChartDete:

    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        import os

        os.makedirs("huggingface")

        snapshot_download("tdsone/chartdete", local_dir="huggingface")

    @enter()
    def enter(self):
        # predict bounding boxes for labels
        from mmdet.apis import init_detector

        # Specify the path to model config and checkpoint file
        config_file = "/root/huggingface/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py"
        checkpoint_file = "/root/huggingface/checkpoint.pth"

        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device="cuda:0")

    def _inference(self, run_id: str):
        print("Running ChartDete...")

        from pathlib import Path
        from mmdet.apis import inference_detector
        import os

        vol.reload()

        inputs = os.listdir(f"/data/{run_id}/input")

        img_paths = [
            os.path.join(Path(f"/data/{run_id}/input"), Path(img)) for img in inputs
        ]

        results_base_folders = [
            Path(f"/data/{run_id}/output/{img}/chartdete") for img in inputs
        ]

        for img_path, results_base_folder in zip(img_paths, results_base_folders):
            try:

                predictions = inference_detector(self.model, img_path)

                result_path = f"{results_base_folder}/predictions.jpg"

                self.model.show_result(
                    img_path,
                    predictions,
                    out_file=result_path,
                )

                vol.commit()

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

                # save coordinates to json
                with open(f"{results_base_folder}/bounding_boxes.json", "w") as f:
                    import json

                    json.dump(results_labelled, f)
                    vol.commit()

                import cv2
                import numpy as np

                # Define the bounding box data
                bounding_boxes = results_labelled

                # Confidence threshold
                confidence_threshold = 0.9

                # Load the image
                image = cv2.imread(img_path)

                if image is None:
                    print("Error loading image")
                    exit()

                label_coordinates = {}

                plot_areas = sorted(
                    bounding_boxes["plot_area"], key=lambda el: el[4], reverse=True
                )

                highest_conf_pa = plot_areas[0]

                label_coordinates["plot_area"] = highest_conf_pa

                # Function to crop bounding boxes
                def crop_bounding_boxes(image, boxes, threshold):
                    cropped_images = []
                    for box in boxes:
                        x1, y1, x2, y2, confidence = box
                        if confidence >= threshold:
                            cropped_image = image[int(y1) : int(y2), int(x1) : int(x2)]
                            cropped_images.append(cropped_image)
                    return cropped_images

                # Crop bounding boxes from both 'x_title' and 'y_title'
                cropped_x_labels = crop_bounding_boxes(
                    image, bounding_boxes["xlabel"], confidence_threshold
                )
                cropped_y_labels = crop_bounding_boxes(
                    image, bounding_boxes["ylabel"], confidence_threshold
                )

                # Save cropped images
                for i, cropped_image in enumerate(cropped_x_labels):
                    path = os.path.join(
                        results_base_folder, Path(f"cropped_xlabels_{i}.jpg")
                    )
                    cv2.imwrite(
                        path,
                        cropped_image,
                    )

                    label_coordinates[path] = bounding_boxes["xlabel"][i]
                    vol.commit()

                for i, cropped_image in enumerate(cropped_y_labels):
                    path = os.path.join(
                        results_base_folder, Path(f"cropped_ylabels_{i}.jpg")
                    )
                    cv2.imwrite(
                        path,
                        cropped_image,
                    )
                    label_coordinates[path] = bounding_boxes["ylabel"][i]
                    vol.commit()

                with open(f"{results_base_folder}/label_coordinates.json", "w") as f:
                    json.dump(label_coordinates, f)
                    vol.commit()

                print("Cropping completed and images saved.")
            except Exception as e:
                print(e)

    @method()
    def inference(self, run_id: str):
        self._inference(run_id)
