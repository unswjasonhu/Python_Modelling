#!/usr/bin/python
from __future__ import division

#enable traceback error report
import cgi, cgitb; cgitb.enable()

import json
import sys, os, inspect

import MySQLdb

# realpath() will make your script run, even if you symlink it :) 
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))

#print cmd_folder
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"comp4335")))
if cmd_subfolder not in sys.path:
    #print cmd_subfolder
    sys.path.insert(0, cmd_subfolder)

#remember to update this appropriately
from resources import get_flattened_index, get_coords_sydney
import numpy as np

[NW_BOUND,SW_BOUND,NE_BOUND,SE_BOUND] = [(-33.728545, 150.849275), (-33.982863, 150.849275), (-33.728545, 151.24753), (-33.982863, 151.24753)]


estimates_table = "Estimates_zeroMean"

# store form fields
form = cgi.FieldStorage()
input_datetime = form.getvalue('input_datetime')
input_date = form.getvalue('input_date')

timeseries = False

if form.getvalue('lat') is not None or form.getvalue('lat') is not None:
    lat = float(form.getvalue('lat'))
    lon = float(form.getvalue('lon'))
    timeseries = True
db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )

# prepare a cursor object using cursor() method
cursor = db.cursor()
# return a grid back
if not timeseries:
    #http://162.222.176.235/cgi-bin/get_estimates_data.py?input_datetime=2015-09-10%2010:00:00
    #get the coords of sydney, correct and offset lat/lon to centre of each square in grid, for google maps
    coords = np.array(get_coords_sydney(centre_offset=False))
    coords = coords.reshape(10000,2)
    coords = coords.tolist()
    sql_str = """select grid_location, co from {0} where datetime="{1}" order by grid_location;""".format(estimates_table, input_datetime)
    cursor.execute(sql_str)
    results = cursor.fetchall()
    results = [(row[0], float(row[1])) for row in results]
    results = zip(coords, results)
    #print results
    results = {input_datetime : results}
elif input_datetime is None :
    # e.g. http://162.222.176.23/cgi-bin/get_estimates_data.py?input_date=2015-09-10&lat=-33.92313&lon=150.98812
    #TODO - if this is slow, it's an index problem
    location = get_flattened_index(lat, lon)

    sql_str = """select time, co from {0} where date="{1}" and grid_location={2} order by datetime;""".format(estimates_table, input_date, location)
    cursor.execute(sql_str)
    results = cursor.fetchall()
    results = {input_date : [(row[0], float(row[1])) for row in results], "location": location}
else:
    results = {"error":"incorrect input provided"}

#print input_datetime
#if input_datetime is None:
#    input_datetime = "2015-09-03 10:00:00"

#if debug:
#    input_date = "2015-08-05"
#    lat = -33.92313
#    lon = 150.98812
#    location = get_flattened_index(lat, lon)
    #print location

# print results
body = json.dumps(results)


print "Content-Type: application/json"
print "Status: 200 OK"
print 
print ""
print body

