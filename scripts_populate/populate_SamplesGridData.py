#! /usr/bin/python
from __future__ import division


import MySQLdb
import sys
from datetime import timedelta, datetime
import numpy as np

from ..resources import data_from_db, gridify_sydney
from ..resources import NW_BOUND,SW_BOUND,NE_BOUND, create_mean_value_grid, get_season

import pdb

def main(granularity):
    """

    SVM uses the known data, called SamplesGridData, training the model with this data
    and using the resultant model to infer unknown data points. There is no interpolation
    in this method.. 

    """

    # Open database connection
    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )

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

    non_zero_grid_count_threshold = 10

    #second pass is needed for inputting the mean and stddev for all the rows of the table
    populate_initially = True
    populate_second_pass = True

    data_table = granularity['data_table']
    interval_period = granularity['interval_period']
    date_format = granularity["date_format"]
    epoch_variable = granularity['epoch_variable']

    #epochs are time period to iterate over
    #provide some buffer time (extra epoch)
    epochs =  ((end_date - start_date).days*epoch_variable + 1)
    
    print "Start data and end date: {0} to {1}".format(start_date, end_date)
    print "Number of hours of data: {0}".format(epochs)
    
    target_date = start_date
    total_rows = 0
    no_epoch_count = 0
    skip_epoch_count = 0

    #populate the first stage of the process for samplesGridData
    if populate_initially:
        for _ in xrange(epochs):
            #do a quick check to see if data for a datetime exists, skip if it does
            sql_str = """select datetime from {0} where datetime="{1}" limit 1;""".format(data_table, target_date)
            cursor.execute(sql_str)
            if cursor.rowcount > 0:
                skip_epoch_count += 1
                target_date += timedelta(seconds=interval_period)
                continue

            #is there sensor data, skip if not
            select_str = """select 
                                * 
                            from 
                                Samples 
                            where user_id != 2 and date like "{0}%" and co < 60 and co > 0 
                                limit 1;""".format(target_date.strftime(date_format))
            cursor.execute(select_str)
            if cursor.rowcount == 0:
                skip_epoch_count += 1
                print "Skipped {0} due to lack of sensor data".format(target_date)
                target_date += timedelta(seconds=interval_period)
                continue;

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
                                    target_date, 
                                    NW_BOUND[0], SW_BOUND[0], NW_BOUND[1], NE_BOUND[1], 
                                    interval_period
                                )

            df_mysql = data_from_db(select_str, verbose=True, exit_on_zero=False)
            if df_mysql is None:
                print "No data returned for {0}".format(target_date)
                no_epoch_count += 1
                target_date += timedelta(seconds=interval_period)
                continue

            #check the number of bins or grid locations populated
            _, non_zero_grid_count = create_mean_value_grid(df_mysql)

            #discount grid if it doesn't have enough pixels (i.e. less than threshold)
            if non_zero_grid_count < non_zero_grid_count_threshold:
                skip_epoch_count += 1
                print "Skipped {0} due to non zero grid count less than threshold".format(target_date)
                target_date += timedelta(seconds=interval_period)
                continue

            #interpolate to get a grid
            known, z, ask, _ = gridify_sydney(df_mysql, verbose=False, heatmap=False)
            
            if len(known) == 0:
                raise Exception("No data for {0}".format(target_date))
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
                print "error: 4 fixed station values not found for {}".format(target_date)
                sys.exit()

            FIXED_LOCATIONS = ['Chullora', 'Liverpool', 'Prospect', 'Rozelle']
            mean_fixed = np.nanmean([fixed_samples_data[fixed_samples_data.location_name==location]['co'].iloc[0] for location in FIXED_LOCATIONS])

            co_chullora = fixed_samples_data[fixed_samples_data.location_name=='Chullora']['co'].iloc[0] if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Chullora']['co'].iloc[0]) else mean_fixed
            co_liverpool = fixed_samples_data[fixed_samples_data.location_name=='Liverpool']['co'].iloc[0] if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Liverpool']['co'].iloc[0]) else mean_fixed
            co_prospect = fixed_samples_data[fixed_samples_data.location_name=='Prospect']['co'].iloc[0]  if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Prospect']['co'].iloc[0]) else mean_fixed
            co_rozelle = fixed_samples_data[fixed_samples_data.location_name=='Rozelle']['co'].iloc[0]  if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Rozelle']['co'].iloc[0]) else mean_fixed

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
            
            print "At {0}, Number of rows considered in total: {1}".format(target_date, total_rows)
            # commit
            db.commit()
            target_date += timedelta(seconds=interval_period)

    # after all the rows have been populated with the original co, 
    # we need to populate the normalised value, mean and std
    if populate_second_pass:
        select_str = """ select * from {};""".format(data_table)
        df_mysql = data_from_db(select_str, verbose=True, exit_on_zero=False)
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
    print "No epoch count: {0} and Skip epoch counts {1}".format(no_epoch_count,skip_epoch_count)


if __name__ == "__main__":
        print "Starting script"
        start_time = datetime.now()
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
                "data_table": "samplesGridData",
            },
        }

        main(granularities['hour'])
        end_time = datetime.now()
        time_taken = end_time - start_time
        print "Time taken is ", time_taken.seconds
        print "Script finished!"

