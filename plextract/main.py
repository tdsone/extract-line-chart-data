from modal import Cls, Mount

from plextract.modal import vol, app


@app.function(
    volumes={"/data": vol},
    mounts=[Mount.from_local_dir("input", remote_path="/input")]
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

    import concurrent.futures

    LineFormer = Cls.lookup("plextract-lineformer", "LineFormer")
    ChartDete = Cls.lookup("plextract-chartdete", "ChartDete")

    os.makedirs(f"/data/{run_id}/predictions/ocr")
    os.makedirs(f"/data/{run_id}/predictions/chartdete")

    vol.commit()

    vol.reload()

    lineformer_preds = LineFormer().inference.remote(run_id)
    chartdete_preds = ChartDete().inference.remote(run_id)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # Submit tasks to the executor
    #     future1 = executor.submit(LineFormer().inference.remote, run_id)
    #     print("Started LineFormer...")
    #     future2 = executor.submit(ChartDete().inference.remote, run_id)
    #     print("Started ChartDete...")

    #     # Collect results
    #     lineformer_preds = future1.result()
    #     chartdete_preds = future2.result()

    # OCR text from axis labels
    vol.reload()

    axis_label_images = [
        file
        for file in os.listdir(f"/data/{run_id}/predictions/chartdete")
        if "label" in file
    ]

    print("Found axis label images:")
    print(axis_label_images)

    OCRModel = Cls.lookup("plextract-ocr", "OCRModel")

    label_texts = list(OCRModel().inference.map(axis_label_images))

    return (lineformer_preds, chartdete_preds)

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
