import json
import os
from typing import Dict, Any
from PIL.Image import Image

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, IMAG_PATH, DATA_DIR, NAME, CROP_POLYGON_OVERLAP, RED

from pandas import DataFrame


def make_crop(coordinates: tuple[float, float], path: str, color: str, diameter: float):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    'cropped_img' the cropped image based on the coordinates
    """
    img = Image.open(path)
    x, y = coordinates
    inner_padding_pixels = diameter / 5  # area between light bulb to frame
    outer_padding_pixels = diameter / 3  # area between the light bulbs
    height = 3 * diameter + outer_padding_pixels * 2 + inner_padding_pixels * 2
    width = height / 3
    if color == RED:
        y = y - (diameter / 2 + outer_padding_pixels)
    else:
        y = y + (diameter / 2 + outer_padding_pixels) - height

    x0 = x - width / 2
    x1 = x + width / 2
    y1 = y + height
    cropped_img = img.crop((x0, y, x1, y1))
    return x0, x1, y, y1, cropped_img


def check_crop(img_path, x0, x1, y0, y1, center_x, center_y, threshold=CROP_POLYGON_OVERLAP):
    is_traffic_light, ignore_crop = False
    json_path = img_path.replace('leftImg8bit.png', 'gtFine_polygons.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
        for obj in data['objects']:
            if obj['label'] == 'traffic light':
                polygon = obj['polygon']
                x_coords, y_coords = zip(*polygon)
                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)
                if (min_x <= center_x <= max_x) and (
                        min_y <= center_y <= max_y):  # check that  center point is contained in the traffic light
                    overlap_x0 = max(x0, min_x)
                    overlap_x1 = min(x1, max_x)
                    overlap_y0 = max(y0, min_y)
                    overlap_y1 = min(y1, max_y)

                    if overlap_x0 < overlap_x1 and \
                            overlap_y0 < overlap_y1:  # check that crop captures "threshold" % of the polygon
                        overlap_area = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
                        tl_box_area = (max_x - min_x) * (max_y - min_y)
                        dimension = overlap_area / tl_box_area
                        if dimension >= threshold or dimension <= (1 - CROP_POLYGON_OVERLAP):
                            return True, False
                        else:
                            return True, True
    return is_traffic_light, ignore_crop


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You want stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    crop_counter = 0
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        x0, x1, y0, y1, crop = make_crop(df[X], df[Y], df[IMAG_PATH], df[COLOR])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = os.path.join(DATA_DIR, CROP_DIR, df[NAME], df[COLOR], crop_counter)
        crop_counter += 1
        crop.save(CROP_DIR / crop_path)
        result_template[CROP_PATH] = crop_path
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(df[IMAG_PATH], x0, x1, y0, y1, df[X], df[Y])

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)

    return result_df
