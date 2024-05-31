from modal import Mount
from plextract.run_lineformer import Model as LineFormer
from plextract.modal import vol, app


@app.function(
    volumes={"/predictions": vol},
)
def run_pipeline():
    import os
    import uuid

    run_id = str(uuid.uuid4())

    print(f"Processing run {run_id}...")

    predictions = LineFormer().inference.remote(run_id=run_id)

    return predictions

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
