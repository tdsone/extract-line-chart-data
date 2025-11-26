from modal import Image

from .modal import vol, modal_app
from ..utils import correct_coordinates

from .chartdete import ChartDete
from .lineformer import LineFormer
from .trocr import OCRModel

image = (
    Image.debian_slim()
    .pip_install("scipy", "matplotlib")
)


@modal_app.function(image=image, volumes={"/data": vol})
def run_pipeline(input_dir, output_dir, run_id):
    import os
    from ..utils import logger

    print(f"Processing run with run id: {run_id}...")

    BASE_INPUT = f"/data/{input_dir}" 
    BASE_OUTPUT = f"/data/{output_dir}"

    os.makedirs(BASE_OUTPUT)
    vol.commit()
    vol.reload()

    input_files = os.listdir(BASE_INPUT) # input_dir is a remote dir

    logger.debug(input_files)

    print("Creating output folders...")
    for file in input_files:
        file_dir = f"{BASE_OUTPUT}/{file}"
        os.makedirs(f"{file_dir}/chartdete")
        os.makedirs(f"{file_dir}/lineformer")

    vol.commit()

    vol.reload()

    print("Extracting lines from images...")
    tasks = [(run_id, img) for img in input_files]
    lineformer_infer = LineFormer().inference
    _ = list(
        lineformer_infer.starmap(
            input_iterator=tasks, return_exceptions=True, wrap_returned_exceptions=False
        )
    )
    print("Detecting chart elements...")
    chartdete_preds = ChartDete().inference.remote(run_id)

    # OCR text from axis labels
    vol.reload()

    axis_label_images = []
    for plot_img_dir in os.listdir(BASE_OUTPUT):
        chartdete_dir = f"{BASE_OUTPUT}/{plot_img_dir}/chartdete"

        for label_img in os.listdir(chartdete_dir):
            if "label" in label_img and ".json" not in label_img:
                axis_label_images.append(f"{chartdete_dir}/{label_img}")

    label_texts = list(
        OCRModel().inference.map(axis_label_images, return_exceptions=True)
    )

    # Save ocrred values to file
    for img_dir in os.listdir(f"{BASE_OUTPUT}"):
        path = f"{BASE_OUTPUT}/{img_dir}/axis_label_texts.json"
        print(f"Saving ocr results to {path}...")
        with open(path, "w") as f:
            import json

            json.dump(
                {
                    path: extracted_text
                    for path, extracted_text in label_texts
                    if img_dir in path
                },
                f,
            )
            vol.commit()

    vol.reload()

    print("Correcting coordinates...")
    for img in os.listdir(BASE_INPUT):
        correct_coordinates(BASE_OUTPUT, img)

    vol.commit()

    return chartdete_preds


@modal_app.local_entrypoint()
def main():
    predictions = run_pipeline.remote()

    print(predictions)
