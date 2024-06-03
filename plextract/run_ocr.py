"""
This script runs OCR using Microsofts trocr-base-handwritten on the axis label images.
"""

from modal import App, Image, build, enter, method
from pathlib import Path

from plextract.modal import vol

app = App("plextract-ocr")

ocr_img = Image.debian_slim().pip_install("transformers", "pillow", "torch")

with ocr_img.imports():
    from PIL import Image, ImageOps
    import io
    import json
    import os
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# @app.function(image=ocr_img, secrets=[Secret.from_name("gcp-twig")])
# def run_ocr(blob_name):

#     print(f"Running OCR on {blob_name}")

#     # Download the image as a blob
#     blob = bucket.blob(blob_name)
#     # Download the image to a file
#     blob.download_to_filename("/tmp/temp_image")

#     # verify that the image has more than 0 bytes
#     if os.path.getsize("/tmp/temp_image") == 0:
#         print(f"Image is empty: {blob_name}")
#         return

#     # Open the image file
#     image = Image.open("/tmp/temp_image")

#     pixel_values = processor(image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)

#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     return blob_name, generated_text


@app.cls(
    image=ocr_img,
    volumes={"/data": vol},
)
class OCRModel:

    @build()
    def build(self):
        pass

    @enter()
    def enter(self):

        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        pass

    @method()
    def inference(self, path: Path):

        # Make sure the file to make the prediction on is there
        vol.reload()

        # Open the image file
        image = Image.open(path)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text


# @app.function()
# def extract_text(run_id: str) -> None:

#     results = list(
#         run_ocr.map(
#             [blob.name for blob in blobs if "plot_area" not in blob.name],
#             return_exceptions=True,
#         )
#     )

#     for el in results:
#         try:
#             blob_name, ocr = el
#             ocrs[blob_name] = ocr
#         except Exception as e:
#             print(f"Error processing blob: {el}")

#     # upload ocrs to gcp bucket
#     ocr_blob = bucket.blob("ocr.json")
#     ocr_blob.upload_from_string(json.dumps(ocrs))
