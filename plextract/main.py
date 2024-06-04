from modal import Cls, Image, Mount

from plextract.modal import vol, app
from plextract.correct_coordinates import correct_coordinates

image = Image.debian_slim().pip_install("scipy", "matplotlib")


@app.function(
    image=image,
    volumes={"/data": vol},
    mounts=[Mount.from_local_dir("input", remote_path="/input")],
)
def run_pipeline():
    import os
    import shutil
    import uuid

    run_id = str(uuid.uuid4())

    print(f"Processing run with run id: {run_id}...")

    BASE_INPUT = f"/data/{run_id}/input"
    BASE_OUTPUT = f"/data/{run_id}/output"

    # copy mounted input to volume
    # (necessary as run id not known at build time)
    os.makedirs(BASE_INPUT)

    print("Copying files from mount to vol...")
    input_files = os.listdir("/input")

    for file_name in input_files:
        shutil.copy(os.path.join("/input", file_name), BASE_INPUT)
        vol.commit()

    LineFormer = Cls.lookup("plextract-lineformer", "LineFormer")
    ChartDete = Cls.lookup("plextract-chartdete", "ChartDete")

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
    lineformer_preds = list(
        lineformer_infer.starmap(input_iterator=tasks, return_exceptions=True)
    )
    print("Detecting chart elements...")
    chartdete_preds = ChartDete().inference.remote(run_id)

    # OCR text from axis labels
    vol.reload()

    from pathlib import Path

    axis_label_images = []
    for plot_img_dir in os.listdir(BASE_OUTPUT):
        chartdete_dir = f"{BASE_OUTPUT}/{plot_img_dir}/chartdete"

        for label_img in os.listdir(chartdete_dir):
            if "label" in label_img and ".json" not in label_img:
                axis_label_images.append(f"{chartdete_dir}/{label_img}")

    OCRModel = Cls.lookup("plextract-ocr", "OCRModel")

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
        correct_coordinates(run_id, img)

    vol.commit()

    return chartdete_preds

    # dir = Path("/tmp/stable-diffusion-xl")
    # if not dir.exists():
    #     dir.mkdir(exist_ok=True, parents=True)

    # output_path = dir / "output.png"
    # print(f"Saving it to {output_path}")
    # with open(output_path, "wb") as f:
    #     f.write(image_bytes)

    pass


@app.local_entrypoint()
def main():
    predictions = run_pipeline.remote()

    print(predictions)
