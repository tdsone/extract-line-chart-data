"""
This script runs OCR using Microsofts trocr-base-handwritten on the axis label images.

Run it with: 

modal run ocr.py
"""

import modal

app = modal.App("ocr")

tesseract_img = modal.Image.debian_slim().pip_install(
    "google-cloud-storage", "transformers", "pillow", "torch"
)

SECRET_NAME = ""

with tesseract_img.imports():
    from google.cloud import storage
    from google.oauth2 import service_account
    from PIL import Image, ImageOps
    import io
    import json
    import os
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    client = storage.Client(credentials=credentials)
    bucket_name = "plot-parts-cropped"  # Replace with your bucket name
    bucket = client.get_bucket(bucket_name)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )


@app.function(image=tesseract_img, secrets=[modal.Secret.from_name("gcp-twig")])
def run_ocr(blob_name):

    print(f"Running OCR on {blob_name}")

    # Download the image as a blob
    blob = bucket.blob(blob_name)
    # Download the image to a file
    blob.download_to_filename("/tmp/temp_image")

    # verify that the image has more than 0 bytes
    if os.path.getsize("/tmp/temp_image") == 0:
        print(f"Image is empty: {blob_name}")
        return

    # Open the image file
    image = Image.open("/tmp/temp_image")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return blob_name, generated_text


@app.function(image=tesseract_img, secrets=[modal.Secret.from_name(SECRET_NAME)])
def extract_text():

    ocrs = {}

    # List all objects in the bucket
    blobs = list(bucket.list_blobs())

    results = list(
        run_ocr.map(
            [blob.name for blob in blobs if "plot_area" not in blob.name],
            return_exceptions=True,
        )
    )

    for el in results:
        try:
            blob_name, ocr = el
            ocrs[blob_name] = ocr
        except Exception as e:
            print(f"Error processing blob: {el}")

    # upload ocrs to gcp bucket
    ocr_blob = bucket.blob("ocr.json")
    ocr_blob.upload_from_string(json.dumps(ocrs))


@app.local_entrypoint()
def main():
    extract_text.remote()
    pass
