from modal import Cls, Mount

from plextract.modal import vol, app
from plextract.correct_coordinates import correct_coordinates


@app.function(
    volumes={"/data": vol}, mounts=[Mount.from_local_dir("input", remote_path="/input")]
)
def run_pipeline():
    import os
    import shutil
    import uuid

    run_id = str(uuid.uuid4())

    print(f"Processing run {run_id}...")

    # copy mounted input to volume
    # (necessary as run id not known at build time)
    os.makedirs(f"/data/{run_id}/input")

    for file_name in os.listdir("/input"):
        shutil.copy(os.path.join("/input", file_name), f"/data/{run_id}/input")
        vol.commit()

    LineFormer = Cls.lookup("plextract-lineformer", "LineFormer")
    ChartDete = Cls.lookup("plextract-chartdete", "ChartDete")

    os.makedirs(f"/data/{run_id}/predictions/ocr")
    os.makedirs(f"/data/{run_id}/predictions/chartdete")

    vol.commit()

    vol.reload()

    LineFormer().inference.remote(run_id)
    chartdete_preds = ChartDete().inference.remote(run_id)

    # OCR text from axis labels
    vol.reload()

    from pathlib import Path

    axis_label_images = [
        f"/data/{run_id}/predictions/chartdete/{file}"
        for file in os.listdir(f"/data/{run_id}/predictions/chartdete")
        if "label" in file
    ]

    print("Found axis label images:")
    print(axis_label_images)

    OCRModel = Cls.lookup("plextract-ocr", "OCRModel")

    label_texts = list(
        OCRModel().inference.map(axis_label_images, return_exceptions=True)
    )

    # Save ocrred values to file
    with open(f"/data/{run_id}/predictions/ocr/label_texts.json", "w") as f:
        import json

        json.dump({path: extracted_text for path, extracted_text in label_texts}, f)
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
