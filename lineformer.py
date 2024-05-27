import os
import json
from tqdm.notebook import tqdm

import infer
import cv2
import line_utils


CKPT = "iter_3000.pth"
CONFIG = "lineformer_swin_t_config.py"
DEVICE = "cuda:0"

infer.load_model(CONFIG, CKPT, DEVICE)


def predict(img_path):
    img = cv2.imread(img_path)  # BGR format
    line_dataseries = infer.get_dataseries(img, to_clean=False)

    # Visualize extracted line keypoints
    img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))

    # Construct the new path to save the image
    relative_path = os.path.relpath(img_path, BASE_FOLDER)
    new_img_path = os.path.join("predictions", relative_path)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

    cv2.imwrite(new_img_path, img)

    print(new_img_path)

    return line_dataseries


# Ensure the directory for predictions exists
os.makedirs("predictions", exist_ok=True)

# Path to the predictions file
predictions_path = "predictions/line_series.json"

# Initialize the predictions dictionary
if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
    with open(predictions_path, "r") as f:
        preds = json.load(f)
else:
    preds = {}

# Base folder for test plots
BASE_FOLDER = "test_plots/cropped"

# Collect all plot_area_0.jpeg files
files = []
for root, dirs, filenames in os.walk(BASE_FOLDER):
    for filename in filenames:
        if filename == "plot_area_0.jpeg":
            files.append(os.path.relpath(os.path.join(root, filename), BASE_FOLDER))

print(files[:4])

# Process each file and update/save predictions after every prediction
for file in tqdm(files):
    file_key = os.path.basename(os.path.dirname(file)) + ".jpeg"

    if file_key in preds:
        print(f"Skipping file {file_key}")
        continue

    print(f"Processing {file}...")
    # Assuming `predict` is a defined function
    line_series = predict(os.path.join(BASE_FOLDER, file))
    preds[file_key] = line_series

    # Save the updated predictions to the file after each iteration
    with open(predictions_path, "w") as f:
        json.dump(preds, f, indent=4)
