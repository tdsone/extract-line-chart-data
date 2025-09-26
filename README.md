# Extract Line Chart Data

![Example Output](example/plextract.png)
A repo that shows how to automatically extract the data of a line chart. Mainly a wrapper around [LineFormer](https://github.com/TheJaeLal/LineFormer) and [ChartDete](https://github.com/pengyu965/ChartDete/).

## Installation

1. You need a [modal.com](https://modal.com) account to run this repo out of the box. Sign up [here](https://modal.com/signup).
2. Deploy the relevant functions by running: `chmod +x deploy.sh && ./deploy.sh`

If you'd like to see a "modal-free" version of this, ping me.

## Usage

All images in the folder `input` will be processed.

1. Add your images to the `input` folder.
2. In the root folder, run the data extraction using: `modal run main.py`
3. Download the processed files using `modal volume get plextract-vol <run_id>`. The run id is a uuid and can be found in the console log. For the example files, the result will look like this:

   ```
   <run_id>/
   ├── input
   │   ├── input1.jpeg
   │   ├── input2.jpeg
   │   └── input3.png
   └── output
       ├── input1.jpeg
       │   ├── axis_label_texts.json # Text extracted from axis labels
       │   ├── chartdete
       │   │   ├── bounding_boxes.json
       │   │   ├── cropped_xlabels_0.jpg # Cropped images of axis labels
       │   │   ├── ...
       │   │   ├── cropped_ylabels_0.jpg
       │   │   ├── ...
       │   │   ├── label_coordinates.json # Coordinates of the detected elements
       │   │   └── predictions.jpg # Image with bounding boxes of detected elements
       │   ├── converted_datapoints
       │   │   ├── data.json # The extracted data!
       │   │   └── plot.png # The plot generated from the extracted data
       │   └── lineformer
       │       ├── coordinates.json # The image relative coordinates of the lines
       │       └── prediction.png
       ├── input2.jpeg
       │   ├── ...
       └── input3.png
           ├── ...

   14 directories, 60 files
   ```

4. The extracted data is provided as json: e.g. `<run_id>/output/input1.jpeg/converted_datapoints/data.json`.
5. You can use [display_extracted_data.ipynb](display_extracted_data.ipynb) to plot the extracted data.

## How It Works

The pipeline works as follows:

1. Use ChartDete to detect chart elements, most importantly axis labels and the plot area.
2. OCR the numbers from the labels.
3. Extract the coordinates of the lines in the line chart using LineFormer.
4. Correct the coordinates of the lines to be relative to the plot origin.
5. Calculate the conversion from pixels to axis values.
6. Convert the coordinates using the conversion parameter from step before.

## Example

### Input

![Example Input](example/input.png)

### Output

This chart was generated using matplotlib using the extracted data (`example/data.json`)
![Example Output](example/output.png)

## Resources

- [LineFormer](https://github.com/TheJaeLal/LineFormer)
- [ChartDete](https://github.com/pengyu965/ChartDete/)

# Contact

If you need help setting this up or would just like to use it, shoot me an email: mail@timonschneider.de
