#! /usr/bin/python
from __future__ import division


import MySQLdb
import sys
from datetime import datetime
import numpy as np

from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))

from resources import data_from_db, gridify_sydney
from resources import NW_BOUND,SW_BOUND,NE_BOUND, create_mean_value_grid, get_season

import pdb

total_rows = 0
no_epoch_count = 0
skip_epoch_count = 0


def get_time_periods_with_sensor_data(start_date, end_date, data_table, interval):
    """ Get the time periods to populate data for """
    time_periods = []

    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )
    cursor = db.cursor()

    # find the set of hours with sensor data
    sensor_sql_str = """select distinct DATE_FORMAT(date, "%Y-%m-%d %H:") from Samples where date between "{0}" and "{1}" and user_id != 2;""".format(start_date, end_date)
    cursor.execute(sensor_sql_str)
    sensor_dates = set([x[0] for x in cursor.fetchall()])

    # find the hours with fixed station data
    fixed_sql_str = """select distinct DATE_FORMAT(date, "%Y-%m-%d %H:") from Samples where date between "{0}" and "{1}" and user_id = 2;""".format(start_date, end_date)
    cursor.execute(fixed_sql_str)
    fixed_dates = set([x[0] for x in cursor.fetchall()])

    # find the hours already in the database
    existing_sql_str = """select distinct datetime from {0} where datetime between "{1}" and "{2}";""".format(data_table, start_date, end_date)
    cursor.execute(existing_sql_str)
    existing_dates = set([x[0] for x in cursor.fetchall()])

    time_periods = list((sensor_dates & fixed_dates) - existing_dates)

    db.close()
    if interval == 'minute':
        return sorted(["{}{:02d}".format(t, i) for t in time_periods for i in xrange(60)])
    else:
        return sorted(["{}00".format(t) for t in time_periods])


def main(granularity, start_date, end_date):
    """

    SVM uses the known data, called SamplesGridData, training the model with this data
    and using the resultant model to infer unknown data points. There is no interpolation
    in this method.. 

    """

    global total_rows, skip_epoch_count, non_zero_grid_count, no_epoch_count

    # Open database connection
    #db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )

    # prepare a cursor object using cursor() method
    #cursor = db.cursor()

    non_zero_grid_count_threshold = 10

    #second pass is needed for inputting the mean and stddev for all the rows of the table
    populate_initially = True
    populate_second_pass = False

    interval = granularity['interval']
    data_table = granularity['data_table']
    interval_period = granularity['interval_period']
    date_format = granularity["date_format"]
    epoch_variable = granularity['epoch_variable']

    target_datetimes = get_time_periods_with_sensor_data(start_date, end_date, data_table, interval)

    #epochs are time period to iterate over
    epochs =  len(target_datetimes)

    print "Start data and end date: {} to {}".format(start_date, end_date)
    print "Number of time periods of data: {}, granularity is {}".format(epochs, interval)

    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )
    cursor = db.cursor()


    #populate the first stage of the process for samplesGridData
    if populate_initially:
        for target_datetime in target_datetimes:
            target_datetime = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M')
            #get data for an time period
            select_str = """SELECT 
                                date as datetime, DATE_FORMAT(date,"%Y-%m-%d") AS date, 
                                DATE_FORMAT(date,"%H") as time, 
                                DATE_FORMAT(date,"%i") as minute, 
                                if(WEEKDAY(date)<5, true, false) AS weekdays, 
                                WEEKDAY(date) AS dayoftheweek, 
                                latitude, longitude, user_id, co 
                            FROM 
                                Samples 
                            WHERE 
                                user_id != 2 AND date between "{0}" AND DATE_ADD("{0}", INTERVAL {5} SECOND) 
                                AND co is not null and latitude is not null and longitude is not null 
                                AND (latitude <= {1} AND latitude >= {2}) 
                                AND (longitude >= {3} AND longitude <= {4}) AND co > 0 AND co < 60
                            ORDER BY
                                date asc """.format(
                                    target_datetime, 
                                    NW_BOUND[0], SW_BOUND[0], NW_BOUND[1], NE_BOUND[1], 
                                    interval_period
                                )

            import ipdb; ipdb.set_trace()
            df_mysql = data_from_db(select_str, verbose=True, exit_on_zero=False)
            if df_mysql is None:
                print "No data returned for {0}".format(target_datetime)
                no_epoch_count += 1
                continue

            #check the number of bins or grid locations populated
            _, non_zero_grid_count = create_mean_value_grid(df_mysql)

            #discount grid if it doesn't have enough pixels (i.e. less than threshold)
            if non_zero_grid_count < non_zero_grid_count_threshold:
                skip_epoch_count += 1
                print "Skipped {0} due to non zero grid count less than threshold".format(target_datetime)
                continue

            #interpolate to get a grid
            known, z, ask, _ = gridify_sydney(df_mysql, verbose=False, heatmap=False)
            
            if len(known) == 0:
                raise Exception("No data for {0}".format(target_datetime))
                sys.exit()

            columns = df_mysql.columns.values
            vals = list(df_mysql.iloc[0])
            row_dict = dict(zip(columns, vals))
            relevant_columns = ['time','weekdays','dayoftheweek']
            data_common = ['"{}"'.format(row_dict['datetime'].strftime("%Y-%m-%d %H:00:00"))] + \
                    ['"{}"'.format(row_dict['date'])] + \
                    ["{}".format(row_dict[col]) for col in relevant_columns] + \
                    ["{}".format(get_season(row_dict['datetime']))]

            # hour always needs to be used here to retrieve fixed station values
            select_str = """select
                              date, location_name, co
                          from
                              Samples
                          where
                              user_id=2 and date="{0}" and
                              (location_name="Prospect" or location_name="Rozelle"
                              or location_name="Liverpool" or location_name="Chullora")
                          order by
                              location_name;""".format(row_dict['datetime'].strftime("%Y-%m-%d %H:00:00"))

            fixed_samples_data = data_from_db(select_str, verbose=False, exit_on_zero=False)

            try:
                assert len(fixed_samples_data) == 4
            except AssertionError:
                print "error: 4 fixed station values not found for {}".format(target_datetime)
                sys.exit()

            FIXED_LOCATIONS = ['Chullora', 'Liverpool', 'Prospect', 'Rozelle']
            mean_fixed = np.nanmean([fixed_samples_data[fixed_samples_data.location_name==location]['co'].iloc[0] for location in FIXED_LOCATIONS])

            co_chullora = fixed_samples_data[fixed_samples_data.location_name=='Chullora']['co'].iloc[0] \
                if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Chullora']['co'].iloc[0]) else mean_fixed
            co_liverpool = fixed_samples_data[fixed_samples_data.location_name=='Liverpool']['co'].iloc[0] \
                if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Liverpool']['co'].iloc[0]) else mean_fixed
            co_prospect = fixed_samples_data[fixed_samples_data.location_name=='Prospect']['co'].iloc[0]  \
                if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Prospect']['co'].iloc[0]) else mean_fixed
            co_rozelle = fixed_samples_data[fixed_samples_data.location_name=='Rozelle']['co'].iloc[0]  \
                if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Rozelle']['co'].iloc[0]) else mean_fixed

            for i, _ in enumerate(z):
                total_rows += 1
                # input data into sql
                grid_location_row, grid_location_col = known[i]
                data = data_common + \
                        ["{0}".format(x) for x in [grid_location_row, grid_location_col, co_chullora, co_liverpool, co_prospect, co_rozelle, z[i]]]
                
                insert_str = """
                                 insert ignore into {0} 
                                     (datetime, date, time, weekdays, 
                                     dayoftheweek, season, 
                                     grid_location_row, 
                                     grid_location_col, 
                                     co_chullora, co_liverpool, co_prospect, co_rozelle, co_original) 
                                 values 
                                     ({1}); 
                             """.format(data_table, ','.join(data))
                try:
                    cursor.execute(insert_str)
                except:
                    print insert_str
                    pdb.set_trace()
            
            print "At {0}, Number of rows considered in total: {1}".format(target_datetime, total_rows)
            # commit
            db.commit()
    print "No epoch count: {0} and Skip epoch counts {1}".format(no_epoch_count,skip_epoch_count)

    # after all the rows have been populated with the original co, 
    # we need to populate the normalised value, mean and std
    if populate_second_pass:
        select_str = """ select * from {};""".format(data_table)
        df_mysql = data_from_db(select_str, verbose=True, exit_on_zero=False)
        if not df_mysql:
            print "no rows in {}. Script completed".format(data_table)
            return 
        co_mean, co_stddev = df_mysql['co_original'].mean(), df_mysql['co_original'].std(ddof=0)
        df_mysql['co_mean'] = co_mean
        df_mysql['co_stddev'] = co_stddev
        df_mysql['co'] = (df_mysql['co_original']-co_mean)/co_stddev

        for index, row in df_mysql.iterrows():
            update_sql = """
            UPDATE 
                {0} 
            SET 
                co={1}, co_mean={2}, co_stddev={3} 
            WHERE 
                datetime='{4}' AND grid_location_row={5} AND grid_location_col={6}
            """.format(data_table, row['co'], row['co_mean'], row['co_stddev'], row['datetime'], row['grid_location_row'], row['grid_location_col'])
            cursor.execute(update_sql)
        db.commit()

    db.close()


if __name__ == "__main__":
    print "Starting script"
    script_start_time = datetime.now()
    #start_date = datetime(2013,4,22)
    #end_date = datetime(2013,4,25)
    start_date = datetime(2013,3,1)
    end_date = datetime(2015,11,1)

    #table which has data inserted for model training
    granularities = {
        "hour": {
            "interval": "hour",
            "epoch_variable": 24,
            "interval_period": 3600,
            "date_format": "%Y-%m-%d %H",
            "data_table": "samplesGridData",
        },
        "minute": {
            "interval": "minute",
            "epoch_variable": 24*60,
            "interval_period": 60,
            "date_format": "%Y-%m-%d %H:%i",
            "data_table": "samplesGridDataMinutes",
        },
    }

    main(granularities['minute'], start_date, end_date)
    script_end_time = datetime.now()
    time_taken = script_end_time - script_start_time

    print "Time taken is ", time_taken.seconds
    print "Script finished!"

