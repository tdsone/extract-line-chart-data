"""
Local TrOCR wrapper for OCR on axis labels.
"""

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..utils import logger


class OCRModel:
    def __init__(self):
        """Initialize TrOCR model."""
        logger.info("Loading TrOCR model...")
        
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        
        logger.info("Successfully loaded TrOCR!")

    def inference(self, path: str) -> tuple[str, str]:
        """
        Run OCR on a single image.
        
        Args:
            path: Path to the image file
            
        Returns:
            Tuple of (path, extracted_text)
        """
        image = Image.open(path)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return path, generated_text

    def inference_batch(self, paths: list[str]) -> list[tuple[str, str]]:
        """
        Run OCR on multiple images.
        
        Args:
            paths: List of paths to image files
            
        Returns:
            List of tuples (path, extracted_text)
        """
        results = []
        for path in paths:
            try:
                result = self.inference(path)
                results.append(result)
            except Exception as e:
                logger.error(f"OCR failed for {path}: {e}")
                results.append((path, ""))
        return results

