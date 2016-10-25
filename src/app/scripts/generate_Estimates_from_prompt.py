#!/usr/bin/python

import pickle
from datetime import datetime
import time

import numpy as np
from src.app.resources import data_from_db, classify_hour, get_season

import sys, os

# Add folder to path
cmd_folder = '/code'
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
from src.config import config

use_hour_simplification_feature = True
svm_estimates_table = "SVMEstimates"

def main():

    if len(sys.argv) < 3:
        try:
            start_date = datetime(2013,5,26,16)
            #load current model
            file_name = "nn_current_model"
            fileObject = open(file_name,'rb')
        except:
            raise Exception("Please look at the variables defined in the script. The date format in '%Y-%m-%d %H:%M:%S', or the right name for the saved model are incorrect")
    else:
        try:
            start_date = time.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S")
            #load current model
            file_name = datetime(sys.argv[2])
            fileObject = open(file_name,'rb')
        except:
            raise Exception("Please enter the date format in '%Y-%m-%d %H:%M:%S', or the right name for the saved model")


    print("Using {}, for date: {}".format(file_name, start_date))
    pipeline = pickle.load(fileObject)

    #get data for model input
    sql_string = 'select date, location_name, if(WEEKDAY(date)<5, true, false) AS weekdays, WEEKDAY(date) AS dayoftheweek, co  from Samples where user_id=2 and date="{0}" and (location_name="Prospect" or location_name="Rozelle" or location_name="Liverpool" or location_name="Chullora") order by location_name;'.format(start_date)

    fixed_samples_data = data_from_db(sql_string, exit_on_zero=False, verbose=False)
    try:
        #assert that more than 4 stations need to be returned
        #sometimes 8 rows are returned (duplicate records..)
        assert fixed_samples_data is not None and len(fixed_samples_data) >= 4
    except AssertionError:
        #print("Assertion on number of rows returned failed")
        raise Exception("No rows on {0}\n".format(start_date));

    try:
        specific_hour = start_date.hour

        if use_hour_simplification_feature:
            hour_feature = classify_hour(specific_hour)
        else:
            hour_feature = specific_hour

        FIXED_LOCATIONS = ['Chullora', 'Liverpool', 'Prospect', 'Rozelle']
        mean_fixed = np.nanmean([fixed_samples_data[fixed_samples_data.location_name==location]['co'].iloc[0] for location in FIXED_LOCATIONS])
        co_chullora = fixed_samples_data[fixed_samples_data.location_name=='Chullora']['co'].iloc[0] if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Chullora']['co'].iloc[0]) else mean_fixed
        co_liverpool = fixed_samples_data[fixed_samples_data.location_name=='Liverpool']['co'].iloc[0] if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Liverpool']['co'].iloc[0]) else mean_fixed
        co_prospect = fixed_samples_data[fixed_samples_data.location_name=='Prospect']['co'].iloc[0]  if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Prospect']['co'].iloc[0]) else mean_fixed
        co_rozelle = fixed_samples_data[fixed_samples_data.location_name=='Rozelle']['co'].iloc[0]  if not np.isnan(fixed_samples_data[fixed_samples_data.location_name=='Rozelle']['co'].iloc[0]) else mean_fixed

        #prepare data to be inserted into svm_estimates table
        data = [fixed_samples_data['weekdays'].iloc[0], hour_feature, get_season(start_date), 0, 0, co_liverpool, co_prospect, co_chullora, co_rozelle]
    except Exception,  ex:
        raise Exception("Error on {}; SQL Statement is {}; Error is str({})\n".format(start_date, sql_string, str(ex)));

    #print X_train
    X = np.float64([data])


    print("date,hour,weekdays,dayoftheweek,season,grid_row,grid_col,co_liverpool,co_prospect,co_chullora,co_rozelle,co")

    #go through 100x100 grid pixels
    for i in range(100):
        for j in range(100):
            X[0][3] = i
            X[0][4] = j
            y_val = pipeline.predict(X)[0]

            insert_data = ['"{0}"'.format(start_date), '"{0}"'.format(start_date.date()), start_date.hour, fixed_samples_data['weekdays'].iloc[0],
                    fixed_samples_data['dayoftheweek'].iloc[0], get_season(start_date), i, j, co_liverpool, co_prospect, co_chullora, co_rozelle, 5.7464*y_val[0]+3.48652]

            print(','.join([ str(x) for x in insert_data]))

if __name__ == "__main__":
    print("Starting script")
    # execute only if run as a script
    main()
    print("Script finished!")
