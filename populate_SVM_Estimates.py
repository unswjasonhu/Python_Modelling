#!/usr/bin/python

import pickle
from datetime import datetime,timedelta

import numpy as np
from resources import data_from_db, classify_hour, get_season

import pdb

import MySQLdb

use_hour_simplification_feature = True
svm_estimates_table = "SVMEstimates"

def main():
    # Open database connection
    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )
    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    #start_date = datetime(2015,5,1)
    start_date = datetime(2015,5,17)
    end_date = datetime(2016,5,2)


    #log file
    log_file_name = "populate_SVM_estimates_{0}_log.txt".format(datetime.now().strftime("%Y-%m-%d %H"))
    logObject = open(log_file_name,'w')
    logObject.write("Start date: {0}, End date: {1}\n".format(start_date, end_date));

    #load current model
    file_name = "svm_current_model"
    fileObject = open(file_name,'rb')
    pipeline = pickle.load(fileObject)

    #variable to store the number of rows committed to the db
    row_count = 0

    #TODO: Make this set difference work
    #precheck to make the query go faster
    sql_string = """select distinct datetime from {0}; """.format(svm_estimates_table)
    cursor.execute(sql_string)
    inserted_datetimes = cursor.fetchall()
    
    total_hours = (end_date - start_date).days*24
    total_datetimes = {start_date + timedelta(seconds=i*3600) for i in xrange(total_hours)}

    remaining_datetimes = sorted(list(total_datetimes - set(inserted_datetimes)))

    #iterate through all the hours
    for start_date in remaining_datetimes:
        #precheck for date and hour in svm_estimates tables
        if start_date in inserted_datetimes or start_date.hour < 8:
            continue

        #get data for model input
        sql_string = 'select date, location_name, if(WEEKDAY(date)<5, true, false) AS weekdays, WEEKDAY(date) AS dayoftheweek, co  from Samples where user_id=2 and date="{0}" and (location_name="Prospect" or location_name="Rozelle" or location_name="Liverpool" or location_name="Chullora") order by location_name;'.format(start_date)

        fixed_samples_data = data_from_db(sql_string, exit_on_zero=False)
        try:
            #assert that more than 4 stations need to be returned
            #sometimes 8 rows are returned (duplicate records..)
            assert fixed_samples_data is not None and len(fixed_samples_data) >= 4
        except AssertionError as aex:
            #print("Assertion on number of rows returned failed")
            logObject.write("No rows on {0}\n".format(start_date));
            continue
            #pdb.set_trace()

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
            logObject.write("Error on {}; SQL Statement is {}; Error is str({})\n".format(start_date, sql_string, str(ex)));
            continue

        #print X_train
        X = np.float64([data])

        #go through 100x100 grid pixels
        for i in xrange(100):
            for j in xrange(100):
                X[0][3] = i 
                X[0][4] = j
                y_val = pipeline.predict(X)[0]

                insert_data = ['"{0}"'.format(start_date), '"{0}"'.format(start_date.date()), start_date.hour, fixed_samples_data['weekdays'].iloc[0], fixed_samples_data['dayoftheweek'].iloc[0], get_season(start_date), i, j, co_liverpool, co_prospect, co_chullora, co_rozelle, 5.7464*y_val+3.48652]

                insert_str = """insert ignore into {0} (datetime, date, time, weekdays, dayoftheweek, season, grid_location_row, grid_location_col, co_chullora, co_liverpool, co_prospect, co_rozelle, co_original) values ({1}); """.format(svm_estimates_table, ','.join([ str(x) for x in insert_data]))
                #print insert_str
                try:
                    cursor.execute(insert_str)
                except:
                    print insert_str
                    pdb.set_trace()
        #commit the db every 10000 rows
        db.commit()
        row_count += 10000
        print("Committed date: {0} with row count: {1}".format(start_date, row_count))
    db.close()
    logObject.close()

if __name__ == "__main__":
    print("Starting script")
    # execute only if run as a script
    main()
    print("Script finished!")
