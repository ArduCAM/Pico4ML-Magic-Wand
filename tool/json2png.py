import argparse
import glob
import json
import math
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

FIXED_POINT = 256


def mul_fp(a, b):
    return (a * b) / FIXED_POINT


def div_fp(a, b):
    if b == 0:
        b = 1
    return (a * FIXED_POINT) / b


def float_to_fp(a):
    return math.floor(a * FIXED_POINT)


def norm_to_coord_fp(a, range_fp, half_size_fp):
    a_fp = float_to_fp(a)
    norm_fp = div_fp(a_fp, range_fp)
    return mul_fp(norm_fp, half_size_fp) + half_size_fp


def round_fp_to_int(a):
    return math.floor((a + (FIXED_POINT / 2)) / FIXED_POINT)


def gate(a, min_, max_):
    if a < min_:
        return min_
    elif a > max_:
        return max_
    else:
        return a


def rasterize_stroke(stroke_points, x_range, y_range, width, height):
    num_channels = 3
    buffer_byte_count = height * width * num_channels
    buffer = bytearray(buffer_byte_count)

    width_fp = width * FIXED_POINT
    height_fp = height * FIXED_POINT
    half_width_fp = width_fp / 2
    half_height_fp = height_fp / 2
    x_range_fp = float_to_fp(x_range)
    y_range_fp = float_to_fp(y_range)

    t_inc_fp = FIXED_POINT / len(stroke_points)

    one_half_fp = FIXED_POINT / 2

    for point_index in range(len(stroke_points) - 1):
        start_point = stroke_points[point_index]
        end_point = stroke_points[point_index + 1]
        start_x_fp = norm_to_coord_fp(start_point["x"], x_range_fp, half_width_fp)
        start_y_fp = norm_to_coord_fp(-start_point["y"], y_range_fp, half_height_fp)
        end_x_fp = norm_to_coord_fp(end_point["x"], x_range_fp, half_width_fp)
        end_y_fp = norm_to_coord_fp(-end_point["y"], y_range_fp, half_height_fp)
        delta_x_fp = end_x_fp - start_x_fp
        delta_y_fp = end_y_fp - start_y_fp

        t_fp = point_index * t_inc_fp
        if t_fp < one_half_fp:
            local_t_fp = div_fp(t_fp, one_half_fp)
            one_minus_t_fp = FIXED_POINT - local_t_fp
            red = round_fp_to_int(one_minus_t_fp * 255)
            green = round_fp_to_int(local_t_fp * 255)
            blue = 0
        else:
            local_t_fp = div_fp(t_fp - one_half_fp, one_half_fp)
            one_minus_t_fp = FIXED_POINT - local_t_fp
            red = 0
            green = round_fp_to_int(one_minus_t_fp * 255)
            blue = round_fp_to_int(local_t_fp * 255)
        red = gate(red, 0, 255)
        green = gate(green, 0, 255)
        blue = gate(blue, 0, 255)

        if abs(delta_x_fp) > abs(delta_y_fp):
            line_length = abs(round_fp_to_int(delta_x_fp))
            if delta_x_fp > 0:
                x_inc_fp = 1 * FIXED_POINT
                y_inc_fp = div_fp(delta_y_fp, delta_x_fp)
            else:
                x_inc_fp = -1 * FIXED_POINT
                y_inc_fp = -div_fp(delta_y_fp, delta_x_fp)
        else:
            line_length = abs(round_fp_to_int(delta_y_fp))
            if delta_y_fp > 0:
                y_inc_fp = 1 * FIXED_POINT
                x_inc_fp = div_fp(delta_x_fp, delta_y_fp)
            else:
                y_inc_fp = -1 * FIXED_POINT
                x_inc_fp = -div_fp(delta_x_fp, delta_y_fp)
        for i in range(int(line_length) + 1):
            x_fp = start_x_fp + (i * x_inc_fp)
            y_fp = start_y_fp + (i * y_inc_fp)
            x = round_fp_to_int(x_fp)
            y = round_fp_to_int(y_fp)
            if (x < 0) or (x >= width) or (y < 0) or (y >= height):
                continue
            buffer_index = (y * width * num_channels) + (x * num_channels)
            buffer[buffer_index + 0] = red
            buffer[buffer_index + 1] = green
            buffer[buffer_index + 2] = blue

    np_buffer = np.frombuffer(buffer, dtype=np.uint8).reshape(
        height, width, num_channels
    )

    return np_buffer


X_RANGE = 0.6
Y_RANGE = 0.6


def ensure_empty_dir(dirname):
    dir_path = Path(dirname)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir()


def augment_points(points, move_range, scale_range, rotate_range):
    move_x = np.random.uniform(low=-move_range, high=move_range)
    move_y = np.random.uniform(low=-move_range, high=move_range)
    scale = np.random.uniform(low=1.0 - scale_range, high=1.0 + scale_range)
    rotate = np.random.uniform(low=-rotate_range, high=rotate_range)

    x_axis_x = math.cos(rotate) * scale
    x_axis_y = math.sin(rotate) * scale

    y_axis_x = -math.sin(rotate) * scale
    y_axis_y = math.cos(rotate) * scale

    new_points = []
    for point in points:
        old_x = point["x"]
        old_y = point["y"]
        new_x = (x_axis_x * old_x) + (x_axis_y * old_y) + move_x
        new_y = (y_axis_x * old_x) + (y_axis_y * old_y) + move_y
        new_points.append({"x": new_x, "y": new_y})

    return new_points


def save_strokes_as_images(strokes, root_folder, width, height, augment_count, split):
    ensure_empty_dir(root_folder)
    labels = set()
    for stroke in strokes:
        labels.add(stroke["label"].lower())

    if split:
        for label in labels:
            label_path = Path(root_folder, label)
            ensure_empty_dir(label_path)
        label_counts = {}
        for stroke in strokes:
            points = stroke["strokePoints"]
            label = stroke["label"].lower()
            if label == "":
                raise Exception(
                    "Missing label for %s:%d" % (stroke["filename"], stroke["index"])
                )
            if label not in label_counts:
                label_counts[label] = 0
            label_count = label_counts[label]
            label_counts[label] += 1
            raster = rasterize_stroke(points, X_RANGE, Y_RANGE, width, height)
            image = Image.fromarray(raster)
            image.save(
                Path(
                    root_folder,
                    label,
                    label + "." + str(label_count) + "." + uuid.uuid4().hex + ".png",
                )
            )
            for i in range(augment_count):
                augmented_points = augment_points(points, 0.1, 0.1, 0.3)
                raster = rasterize_stroke(
                    augmented_points, X_RANGE, Y_RANGE, width, height
                )
                image = Image.fromarray(raster)
                image.save(
                    Path(
                        root_folder,
                        label,
                        label
                        + "."
                        + str(label_count)
                        + "."
                        + uuid.uuid4().hex
                        + ".png",
                    )
                )
    else:
        label_counts = {}
        for stroke in strokes:
            points = stroke["strokePoints"]
            label = stroke["label"].lower()
            if label == "":
                raise Exception(
                    "Missing label for %s:%d" % (stroke["filename"], stroke["index"])
                )
            if label not in label_counts:
                label_counts[label] = 0
            label_count = label_counts[label]
            label_counts[label] += 1
            raster = rasterize_stroke(points, X_RANGE, Y_RANGE, width, height)
            image = Image.fromarray(raster)
            image.save(
                Path(
                    root_folder,
                    label + "." + str(label_count) + "." + uuid.uuid4().hex + ".png",
                )
            )
            for i in range(augment_count):
                augmented_points = augment_points(points, 0.1, 0.1, 0.3)
                raster = rasterize_stroke(
                    augmented_points, X_RANGE, Y_RANGE, width, height
                )
                image = Image.fromarray(raster)
                image.save(
                    Path(
                        root_folder,
                        label
                        + "."
                        + str(label_count)
                        + "."
                        + uuid.uuid4().hex
                        + ".png",
                    )
                )
    return labels


def generate_features(dataset_, split, nums):
    dir_path = Path(dataset_)
    dataset_jsons = dataset_ + "/*.json" if dir_path.is_dir() else dataset_
    strokes = []
    for filename in glob.glob(dataset_jsons):
        with open(filename, "r") as file:
            file_contents = file.read()
        file_data = json.loads(file_contents)
        for stroke in file_data["strokes"]:
            stroke["filename"] = filename
            strokes.append(stroke)

    image_width = 32
    image_height = 32

    shuffled_strokes = strokes
    np.random.shuffle(shuffled_strokes)

    save_strokes_as_images(
        shuffled_strokes, "datasets", image_width, image_height, nums, split
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stroke")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="The directory where the JSON files are located.\n" "Or the JSON file.",
    )
    parser.add_argument(
        "-s",
        "--split",
        action='store_true',
        help="Whether to create folders by labels, default False.",
    )
    parser.add_argument(
        "-n",
        "--nums",
        type=int,
        default=10,
        help="Increase the number of samples by N times, the default N = 10.",
    )

    args = parser.parse_args()

    generate_features(args.dataset, args.split, args.nums)
