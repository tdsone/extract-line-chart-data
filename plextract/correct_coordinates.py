"""
Create pix2val.json (stores, how many value units on a plot one pixel is worth)
{
    "00a0be8f-7654-1014-973d-d584c24ba1bd": {
        "x": 2, // e.g. this would mean that one pixel in x/horizontal direction is worth 2 value units
        "y": 4,
    }
}



Inputs: 
- all axis labels coordinates for a figure
- all corresponding texts
"""

import json

"""
coordinates.json
- coordinates of each annotated part on full figure image

meaning of array elements: x1, y1, x2, y2, confidence
----------------
{
    '<run_id>-<img_name>.jpeg':
        {
            'ylabel_0.jpeg': [
                91.9613265991211,
                120.58985900878906,
                111.07434844970703,
                145.89064025878906,
                0.9691145420074463
            ],
            'ylabel_1.jpeg': [
                93.3066177368164,
                214.49050903320312,
                110.21461486816406,
                239.9881591796875,
            ],
            ...
...
}

"""


"""
ocr.json 
- Text extracted from each image

{
    "00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1/xlabel_0.jpeg": "10",
    "00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1/xlabel_1.jpeg": "5",
    "00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1/xlabel_2.jpeg": "15",
    ...
}
"""


def sort_and_check_labels(
    label_coordinates: dict, axis_label_texts: dict, img_key: str
):
    """
    This function makes sure that we can use the OCRed values by sorting them twice (value and position) and comparing.

    label_coordinates:
    {
        "x_title": [
            [
                441.2204284667969, 505.3034362792969, 584.9779663085938,
                544.3689575195312, 0.9882757663726807
            ],
            ...
        ],
        "y_title": [
            [
                7.234010696411133, 196.8053741455078, 49.4166145324707, 313.7027893066406,
                0.9896954298019409
            ],
            ...
    ...
    }

    axis_label_texts:
    {
        "/data/40482601-23c3-43f8-86fd-49517b9e2a3e/output/input2.jpeg/chartdete/cropped_xlabels_0.jpg": "0.3",
        ...
    }


    """

    label_img_paths = list(axis_label_texts.keys())

    ocr_values = {
        f"{img_key.split('.')[0]}/{file}": float(
            axis_label_texts[f"{img_key.split('.')[0]}/{file}"]
        )
        for file in label_img_paths
        if "plot_area" not in file
    }

    yvalues = {k: v for k, v in ocr_values.items() if "ylabel" in k}
    xvalues = {k: v for k, v in ocr_values.items() if "xlabel" in k}

    sorted_ylabels = sorted(yvalues.items(), key=lambda item: item[1])
    sorted_xlabels = sorted(xvalues.items(), key=lambda item: item[1])

    def get_coord_key(label_key):
        return label_key.split("/")[1]

    y_coords = {k: label_coordinates[get_coord_key(k)] for k, v in yvalues.items()}
    x_coords = {k: label_coordinates[get_coord_key(k)] for k, v in xvalues.items()}

    # Sort the y and x coordinates
    sorted_y_coords = sorted(
        y_coords.items(), key=lambda item: item[1][1], reverse=True
    )  # reverse=True as y direction is top to bottom
    sorted_x_coords = sorted(x_coords.items(), key=lambda item: item[1][0])

    assert [label for label, _ in sorted_xlabels] == [
        coord for coord, _ in sorted_x_coords
    ], "The keys of sorted_xlabels and sorted_x_coords are not in the same order."

    assert [label for label, _ in sorted_ylabels] == [
        coord for coord, _ in sorted_y_coords
    ], "The keys of sorted_ylabels and sorted_y_coords are not in the same order."

    xaggr = {}
    for (k1, v1), (k2, v2) in zip(sorted_x_coords, sorted_xlabels):
        assert k1 == k2
        xaggr[k1] = {"coord": v1, "val": v2}

    yaggr = {}
    for (k1, v1), (k2, v2) in zip(sorted_y_coords, sorted_ylabels):
        assert k1 == k2
        yaggr[k1] = {"coord": v1, "val": v2}

    return {"xs": xaggr, "ys": yaggr}


def calc_conversion(coord_val_map: dict):
    """
     coord_val_map
     {
        'xs': {
            '00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1/xlabel_3.jpeg': {
                'coord': [
                    199.7735137939453,
                    597.3622436523438,
                    220.34738159179688,
                    627.3770141601562,
                    0.9575890898704529
                ],
                'val': 0.0
            },
            ...
        'ys': {
            '00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1/ylabel_2.jpeg': {
                'coord': [
                    94.44598388671875,
                    490.62774658203125,
                    143.36465454101562,
                    520.423828125,
                    0.9801554083824158
                ],
            'val': 0.1
            ...
    }
    """
    from scipy.stats import linregress

    def get_best_fit(coord_map: dict, direction="x"):

        # Extract x coordinates and corresponding values
        points = [
            (v["coord"][0 if direction == "x" else 1], v["val"])
            for k, v in coord_map.items()
        ]

        # Perform linear regression to find the best fit line
        # Unpack list of tuples into two separate tuples
        x_coords, y_vals = zip(*points)
        slope, intercept, r_value, p_value, std_err = linregress(x_coords, y_vals)

        return {"slope": slope, "intercept": intercept}

    xpix2val = get_best_fit(coord_val_map["xs"])
    ypix2val = get_best_fit(coord_val_map["ys"], direction="y")

    return {"x": xpix2val, "y": ypix2val}


"""

Conversions
-----------

00098190-6c58-1014-8ce1-bb13c4d7fce9_486947v1_fig3_1.jpeg

{'x': {'slope': 0.018507530062525264, 'intercept': -3.7094232933948135},
 'y': {'slope': 3.275499693756956, 'intercept': -309.137453308719}}
"""


def convert_data_points(conversions, img):
    # Load the coordinates and line series data

    with open("line_series.json") as f:
        all_lineseries = json.load(f)

    # Extract the plot area coordinates
    plot_area_coords = coordinates[img]["plot_area_0.jpeg"][:4]
    # plot area coords are x1, y1, x2, y2

    # Adjust origin to lower left corner (x1, y2)
    origin = plot_area_coords[0], plot_area_coords[3]

    plot_area_height = plot_area_coords[3] - plot_area_coords[1]
    assert plot_area_height > 0

    def convertx(x):
        return x * conversions["x"]["slope"]

    def converty(y):
        return y * conversions["y"]["slope"] * -1

    # Initialize a dictionary to store the converted line series
    converted_lineseries = {}

    # Loop over all line series
    for series_index, lineseries in enumerate(all_lineseries[img]):

        # Adjust the coordinates such that the line series is in relation to the lower left corner
        converted_points = [
            {"x": convertx(point["x"]), "y": converty(plot_area_height - point["y"])}
            for point in lineseries
        ]

        # Save the converted points into the dictionary with the series index as the key
        converted_lineseries[f"series_{series_index}"] = converted_points

    # Save the converted line series to a JSON file
    import os

    os.makedirs(f"converted_datapoints/{img.split('.')[0]}", exist_ok=True)
    json_filename = f"converted_datapoints/{img.split('.')[0]}/data.json"
    with open(json_filename, "w") as json_file:
        json.dump(converted_lineseries, json_file, indent=4)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot each line series
    for series_name, points in converted_lineseries.items():
        x_vals = [point["x"] for point in points]
        y_vals = [point["y"] for point in points]
        plt.plot(x_vals, y_vals, label=series_name)

    # Label the axes
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    # Add a legend
    plt.legend()

    # Save the plot to a file
    plot_filename = f"converted_datapoints/{img.split('.')[0]}/plot.png"
    plt.savefig(plot_filename)


def correct_coordinates(run_id: str, img: str):

    with open(f"/data/{run_id}/output/{img}/chartdete/coordinates.json", "r") as f:
        label_coordinates = json.load(f)

    with open(f"/data/{run_id}/output/{img}/axis_label_texts.json", "r") as f:
        axis_label_texts = json.load(f)

    try:
        labels = sort_and_check_labels(img)
        conversions = calc_conversion(labels)
        convert_data_points(conversions=conversions, img=img)
    except Exception as e:
        print(f"Correcting coordinates of {img} did not work!")
        print(e)
