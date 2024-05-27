"""
This script extracts data from plots using LineFormer and ChartDete.

If you have not used modal before, you can set up an account here: modal.com

Run this script using: modal run chartdete/main.py
"""

from pathlib import Path
from modal import Image, App
import modal

app = App("extract_data_from_plot")
SECRET_NAME = ""

plot_extr_image = (
    Image.debian_slim(python_version="3.8")
    .apt_install("git")
    .run_commands("git clone https://github.com/pengyu965/ChartDete.git")
    .run_commands("git clone https://github.com/TheJaeLal/LineFormer.git")
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
    .pip_install("bresenham", "tqdm")
)

@app.function(
    image=plot_extr_image,
    timeout=3000,
    gpu="any",
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def extract_data_from_plot(plot_jpg: str, bucket_name: str):
    # download image from google cloud
    import json
    import os

    from google.oauth2 import service_account
    from google.cloud import storage
    from pathlib import Path

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    client = storage.Client(credentials=credentials)

    plot_bucket = client.bucket(bucket_name)

    # download image from bucket
    plot_bucket.blob(plot_jpg).download_to_filename(plot_jpg)

    # predict bounding boxes for labels
    from mmdet.apis import init_detector, inference_detector
    import mmcv

    # Specify the path to model config and checkpoint file
    config_file = "chartdete.py"
    checkpoint_file = "checkpoint.pth"

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    predictions = inference_detector(model, plot_jpg)

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
        results_labelled[label] = res

    import cv2
    import numpy as np

    # Define the bounding box data
    bounding_boxes = results_labelled

    # Confidence threshold
    confidence_threshold = 0.5

    # Load the image
    image = cv2.imread(plot_jpg)

    if image is None:
        print("Error loading image")
        exit()

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
    cropped_x_title = crop_bounding_boxes(
        image, bounding_boxes["ylabel"], confidence_threshold
    )
    cropped_y_title = crop_bounding_boxes(
        image, bounding_boxes["xlabel"], confidence_threshold
    )

    # Save cropped images
    for i, cropped_image in enumerate(cropped_x_title):
        cv2.imwrite(f"cropped_xlabels_{i}.jpg", cropped_image)

    for i, cropped_image in enumerate(cropped_y_title):
        cv2.imwrite(f"cropped_ylabels_{i}.jpg", cropped_image)

    print("Cropping completed and images saved.")

    pass


@app.local_entrypoint()
def main():
    extract_data_from_plot.remote(
        "plot-images-cropped/00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1.jpeg"
    )
