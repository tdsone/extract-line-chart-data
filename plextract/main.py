from modal import Mount
from plextract.run_lineformer import Model as LineFormer
from plextract.modal import vol, app

@app.function(volumes={"/predictions": vol}, mounts=[Mount.from_local_dir("input", remote_path="/input")])
def run_pipeline(): 
    predictions = LineFormer().inference.remote(img_paths=[])

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
    run_pipeline.remote()