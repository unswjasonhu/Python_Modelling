#!/usr/bin/python
from __future__ import division

import sys, os, inspect
import numpy as np
import MySQLdb

# realpath() will make your script run, even if you symlink it :) 
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))

# print cmd_folder
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"comp4335")))
if cmd_subfolder not in sys.path:
    #print cmd_subfolder
    sys.path.insert(0, cmd_subfolder)

# add this if run as a script for resources
if __name__ == "__main__":
    src_folder = os.path.abspath(os.path.join(cmd_folder, os.pardir))
    sys.path.insert(0, src_folder)

from resources import get_index, get_coords_sydney, create_heatmap
# from resources import NW_BOUND,SW_BOUND,NE_BOUND,SE_BOUND

estimates_table = "SVMEstimates"
SERVICES_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMAGES_DIR = os.path.join(SERVICES_DIR, "static/heatmap_images/")


def get_estimates_data_service(input_datetime=None, input_date=None, lat=None, lon=None):
    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    # return a grid back
    if input_datetime and not input_date:
        #http://162.222.176.235/cgi-bin/get_estimates_data.py?input_datetime=2015-09-10%2010:00:00
        #get the coords of sydney, correct and offset lat/lon to centre of each square in grid, for google maps
        coords = np.array(get_coords_sydney(centre_offset=False))
        coords = coords.reshape(10000,2)
        coords = coords.tolist()
        try:
            sql_str = """select grid_location_row, grid_location_col, co_original from {0} where datetime="{1}" order by grid_location_row, grid_location_col;""".format(estimates_table, input_datetime)
            cursor.execute(sql_str)
        except:
            raise Exception("Error in : ", sql_str)

        results = cursor.fetchall()

        if len(results) == 0:
            results = {"error":"no results found for datetime input. Please check your input again"}
        else:
            results = [(row[0], row[1], float(row[2])) for row in results]
            results = zip(coords, results)
            #print results
            results = {input_datetime : results}
    elif input_date and lat and lon:
        #http://162.222.176.235/modeling/get_estimates_data?input_date=2015-09-11&lat=-33.92313&lon=150.98812
        #if this is slow, it's an index problem
        grid_location_row, grid_location_col = get_index(lat, lon)

        try:
            sql_str = """select time, co_original from {} where date="{}" and grid_location_row={} and grid_location_col={} order by datetime;""".format(estimates_table, input_date, grid_location_row, grid_location_col)
            cursor.execute(sql_str)
        except:
            raise Exception("Error in : ", sql_str)

        results = cursor.fetchall()
        results = {input_date : [(row[0], float(row[1])) for row in results], "grid_location_row": grid_location_row, "grid_location_col": grid_location_col}
    else:
        results = {"error":"no output given the provided input parameters. Please check your input again"}

    return results


def generate_2d_plot(input_datetime):
    results = get_estimates_data_service(input_datetime=input_datetime)
    image_name = os.path.join(IMAGES_DIR, input_datetime)

    if not len(results.values()):
        return {"error":"no results given for data. Please check your input again"}

    data = results.values()[0]

    sydney_grid = np.zeros((100,100))

    for _, t in data:
        row, col, val = t
        sydney_grid[row][col] = val

    #delete old image, the dir should have no less than 10
    images = sorted([image for image in os.listdir(IMAGES_DIR) if '.png' in image])
    if len(images) > 10:
        #delete an arbitrary image here
        os.remove(os.path.join(IMAGES_DIR,images[0]))

    #heatmap is created with interpolated values
    create_heatmap(sydney_grid, image_name, strip=True)
    return image_name

if __name__ == "__main__":
    src_folder = os.path.abspath(os.path.join(cmd_folder, os.pardir))
    sys.path.insert(0, src_folder)
    input_datetime="2015-09-10 10:00:00"
    generate_2d_plot(input_datetime)
