from typing import Literal
import uuid

from .utils import logger


def _check_local_deps():
    """Check if local dependencies are installed."""
    try:
        import torch
        import transformers
        import cv2
        return True
    except ImportError:
        return False


def _check_modal_deps():
    """Check if modal dependencies are installed."""
    try:
        import modal
        return True
    except ImportError:
        return False


def extract(
    input_dir: str = "input",
    output_dir: str = "output",
    backend: Literal["local", "modal"] = "local",
):
    match backend:
        case "local":
            if not _check_local_deps():
                raise ImportError(
                    "Local dependencies not installed. "
                    "Install with: pip install plextract[local]"
                )
            logger.info("Running plextract locally...")
            from .local import run_pipeline
            run_pipeline(input_dir, output_dir)
        case "modal":
            if not _check_modal_deps():
                raise ImportError(
                    "Modal dependencies not installed. "
                    "Install with: pip install plextract[modal]"
                )
            logger.info(f"Running plextract remotely on modal for images in \n - {input_dir} \nand saving results to \n - {output_dir}.")
            import modal
            from .modal import run_pipeline, modal_app, vol, download_volume_dir

            """
            We have to upload all images to a modal volume first for further processing.
            """
            run_id = str(uuid.uuid4())
            logger.info(f"Run id is: {run_id}")
            remote_input  = f"{run_id}/input"
            remote_output = f"{run_id}/output"

            with modal.enable_output():
                logger.info("Uploading images to modal volume.")
                with vol.batch_upload(force=True) as batch: 
                    batch.put_directory(input_dir, f"/{remote_input}")
                logger.info("\tDone uploading images to modal volume.")

                logger.info("Running pipeline on modal now.")
                with modal_app.run():
                    run_pipeline.remote(
                        remote_input, 
                        remote_output, 
                        run_id
                    )
                logger.info("\tDone running the modal pipeline.")
                logger.info("Downloading files from modal volume...")


                download_volume_dir(
                    local_dir=output_dir, 
                    remote_dir=remote_output
                )
                
        case _:
            raise Exception(
                f'Unknown option {backend}. The only valid options are: "local", "modal"'
            )
