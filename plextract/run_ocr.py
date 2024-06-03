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

        return path, generated_text