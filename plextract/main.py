from modal import Mount
from plextract.modal import vol, app
from plextract.run_lineformer import LineFormer
from plextract.run_chartdete import ChartDete


@app.function(
    volumes={"/predictions": vol},
)
def run_pipeline():
    import os
    import uuid

    run_id = str(uuid.uuid4())

    print(f"Processing run {run_id}...")

    lineformer_preds = LineFormer().inference.remote(run_id=run_id)
    chartdete_preds = ChartDete().inference.remote(run_id=run_id)

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
