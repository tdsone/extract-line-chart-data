import json

import matplotlib
import pytest

matplotlib.use("Agg")


def test_sort_and_check_labels_returns_sorted_mapping():
    from plextract.utils.correct_coordinates import sort_and_check_labels

    label_coordinates = {
        "sample/xlabel_0.jpeg": [10.0, 50.0, 15.0, 55.0, 0.9],
        "sample/xlabel_1.jpeg": [30.0, 50.0, 35.0, 55.0, 0.9],
        "sample/ylabel_0.jpeg": [5.0, 90.0, 10.0, 95.0, 0.9],
        "sample/ylabel_1.jpeg": [5.0, 10.0, 10.0, 15.0, 0.9],
    }
    axis_label_texts = {
        "sample/xlabel_0.jpeg": "0",
        "sample/xlabel_1.jpeg": "5",
        "sample/ylabel_0.jpeg": "0",
        "sample/ylabel_1.jpeg": "10",
    }

    result = sort_and_check_labels(
        label_coordinates=label_coordinates,
        axis_label_texts=axis_label_texts,
        img_key="sample_image",
    )

    assert list(result["xs"].keys()) == [
        "sample/xlabel_0.jpeg",
        "sample/xlabel_1.jpeg",
    ]
    assert list(result["ys"].keys()) == [
        "sample/ylabel_0.jpeg",
        "sample/ylabel_1.jpeg",
    ]
    assert result["xs"]["sample/xlabel_0.jpeg"]["val"] == 0.0
    assert result["ys"]["sample/ylabel_1.jpeg"]["coord"][1] == pytest.approx(10.0)


def test_calc_conversion_returns_linear_relationship():
    from plextract.utils.correct_coordinates import calc_conversion

    coord_val_map = {
        "xs": {
            "sample/xlabel_0.jpeg": {
                "coord": [10.0, 50.0, 15.0, 55.0, 0.9],
                "val": 0.0,
            },
            "sample/xlabel_1.jpeg": {
                "coord": [30.0, 50.0, 35.0, 55.0, 0.9],
                "val": 5.0,
            },
        },
        "ys": {
            "sample/ylabel_0.jpeg": {
                "coord": [5.0, 90.0, 10.0, 95.0, 0.9],
                "val": 0.0,
            },
            "sample/ylabel_1.jpeg": {
                "coord": [5.0, 10.0, 10.0, 15.0, 0.9],
                "val": 10.0,
            },
        },
    }

    conversions = calc_conversion(coord_val_map)

    assert conversions["x"]["slope"] == pytest.approx(0.25)
    assert conversions["x"]["intercept"] == pytest.approx(-2.5)
    assert conversions["y"]["slope"] == pytest.approx(-0.125)
    assert conversions["y"]["intercept"] == pytest.approx(11.25)


def test_convert_data_points_writes_converted_series(tmp_path):
    from plextract.utils.correct_coordinates import convert_data_points

    base_dir = tmp_path / "output"
    img_name = "figure"
    lineformer_dir = base_dir / img_name / "lineformer"
    lineformer_dir.mkdir(parents=True)

    coordinates_path = lineformer_dir / "coordinates.json"
    with coordinates_path.open("w") as f:
        json.dump([[{"x": 10, "y": 20}, {"x": 50, "y": 80}]], f)

    label_coordinates = {
        "plot_area": [0.0, 0.0, 100.0, 100.0, 1.0],
    }

    conversions = {
        "x": {"slope": 2.0, "intercept": 0.0},
        "y": {"slope": 1.0, "intercept": 0.0},
    }

    convert_data_points(
        conversions=conversions,
        base_output_dir=str(base_dir),
        img=img_name,
        label_coordinates=label_coordinates,
    )

    output_path = base_dir / img_name / "converted_datapoints" / "data.json"
    assert output_path.exists()

    with output_path.open() as f:
        converted = json.load(f)

    assert "series_0" in converted
    assert converted["series_0"] == [
        {"x": 20.0, "y": -80.0},
        {"x": 100.0, "y": -20.0},
    ]

    plot_path = base_dir / img_name / "converted_datapoints" / "plot.png"
    assert plot_path.exists()
