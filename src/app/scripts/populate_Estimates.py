#! /usr/bin/python
from __future__ import division

import MySQLdb
import sys, os, inspect
from datetime import timedelta, datetime
import numpy as np

# Add folder to path
cmd_folder = '/code'
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
from src.config import config

from src.app.resources.resources import data_from_db, gridify_sydney, idw_interpol
from src.app.resources.resources import NW_BOUND,SW_BOUND,NE_BOUND, create_mean_value_grid, get_season

import pdb

def main():
    """

    Populating the Estimates was an initial method used by NN, then by SVM. Here,
    Interpolation was used to calculate values for all the grids. This data was then
    fed into the model to train.

    This technigue could be left for populating the samplesGridData and using the model
    to estimate points. i.e. no interpolation used for estimation

    """

    # Open database connection
    db = MySQLdb.connect(config.DATABASE_URI, config.DATABASE_USER, config.DATABASE_PASSWORD, config.DATABASE_NAME)


    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    # get the oldest date
    #sql_str = """select distinct date from Samples  where user_id = 2 order by date asc limit 1;"""
    # start date
    #cursor.execute(sql_str)
    #start_date = cursor.fetchone()[0]
    start_date = datetime(2013,3,1)

    # get the newest date
    #sql_str = """select distinct date from Samples  where user_id = 2 order by date desc limit 1;"""
    # end date
    #cursor.execute(sql_str)
    #end_date = cursor.fetchone()[0]
    #override
    end_date = datetime(2015,11,1)

    #Choose the data table to use
    zero_mean = True
    non_zero_grid_count_threshold = 10

    #table which has data inserted for model training
    data_table =  "Estimates_zeroMean" if zero_mean else  "Estimates_old"

    #epochs are the time periods to iterate over
    #provide some buffer time (extra epoch)
    epochs =  ((end_date - start_date).days*24 + 1)

    print("Start data and end date: {0} to {1}".format(start_date, end_date))
    print("Number of hours of data: {0}".format(epochs))

    first_date = start_date
    total_rows = 0
    no_epoch_count = 0
    skip_epoch_count = 0

    for _ in range(epochs):
        #do a quick check to see if data for a datetime exists, skip if it does
        sql_str = """ select datetime from {0} where datetime="{1}" limit 1;""".format(data_table, first_date)
        cursor.execute(sql_str)
        if cursor.rowcount > 0:
            skip_epoch_count += 1
            first_date += timedelta(seconds=3600)
            continue

        #is there sensor data, skip if not
        select_str = """select * from Samples where user_id != 2 and date like "{0}%" and co < 60 and co > 0 limit 1;""".format(first_date.strftime("%Y-%m-%d %H"))
        cursor.execute(select_str)
        if cursor.rowcount == 0:
            skip_epoch_count += 1
            print("Skipped {0} due to lack of sensor data".format(first_date))
            first_date += timedelta(seconds=3600)
            continue;

        #get data for an hour
        select_str = """SELECT
                            date as datetime, DATE_FORMAT(date,"%Y-%m-%d") AS date, DATE_FORMAT(date,"%H") as time, if(WEEKDAY(date)<5, true, false) AS weekdays, WEEKDAY(date) AS dayoftheweek, latitude, longitude, user_id, co
                        FROM
                            Samples
                        WHERE
                            user_id != 2 and date between "{0}" and date_add("{0}", interval 1 hour) and co is not null and latitude is not null and longitude is not null AND (latitude <= {1} AND latitude >= {2}) AND (longitude >= {3} AND longitude <= {4}) AND co > 0 AND co < 60
                        ORDER BY
                            date asc """.format(first_date, NW_BOUND[0], SW_BOUND[0], NW_BOUND[1], NE_BOUND[1])
        df_mysql = data_from_db(select_str, verbose=True, exit_on_zero=False)
        if df_mysql is None:
            print("No data returned for {0}".format(first_date))
            no_epoch_count += 1
            first_date += timedelta(seconds=3600)
            continue

        #check the number of bins populated
        _, non_zero_grid_count = create_mean_value_grid(df_mysql)

        #discount grid if it doesn't have enough pixels (i.e. less than threshold)
        if non_zero_grid_count < non_zero_grid_count_threshold:
            skip_epoch_count += 1
            print("Skipped {0} due to non zero grid count less than threshold".format(first_date))
            first_date += timedelta(seconds=3600)
            continue

        #interpolate to get a grid
        known, z, ask, _ = gridify_sydney(df_mysql, verbose=False, heatmap=False)

        if len(known) == 0:
            raise Exception("No data for {0}".format(first_date))
            sys.exit()

        columns = df_mysql.columns.values
        vals = list(df_mysql.iloc[0])
        row_dict = dict(zip(columns, vals))
        relevant_columns = ['time','weekdays','dayoftheweek']
        data_common = ['"{0}"'.format(row_dict['datetime'].strftime("%Y-%m-%d %H:00:00"))] + ['"{0}"'.format(row_dict['date'])] + ["{0}".format(row_dict[col]) for col in relevant_columns] + ["{0}".format(get_season(row_dict['datetime']))]

        if len(known) < 8:
            Nnear = len(known)
        else:
            Nnear = 8

        # do the interpolation
        (interpolation_grid, interpol_name) = idw_interpol(known, z, ask, Nnear=Nnear)

        #implement for zero mean Estimates table
        if zero_mean:
            #do the zero mean bit
            interpolation_grid = interpolation_grid.flatten()
            interpolation_grid = (interpolation_grid - np.mean(interpolation_grid))/np.nanstd(interpolation_grid)

        #add each element to the db as a row
        for i in xrange(len(interpolation_grid)):
            total_rows += 1
            # input data into sql
            data = data_common + ["{0}".format(x) for x in [i, interpolation_grid[i]]]
            #print data
            insert_str = """insert ignore into {0} () values ({1}); """.format(data_table, ','.join(data))
            cursor.execute(insert_str)

        print("At {0}, Number of rows considered in total: {1}".format(first_date, total_rows))
        # commit at each epoch, i.e. every 10000 rows
        db.commit()
        first_date += timedelta(seconds=3600)

    db.close()
    print("No epoch count: {0} and Skip epoch counts {1}".format(no_epoch_count,skip_epoch_count))


if __name__ == "__main__":
        print("Starting script")
        start_time = datetime.now()
        # execute only if run as a script
        main()
        end_time = datetime.now()
        time_taken = end_time - start_time
        print("Time taken is ", time_taken.seconds)
        print("Script finished!")
