#/usr/bin/python
from __future__ import division

import MySQLdb

from resources import *

from datetime import datetime, timedelta

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np

#n nearest values
N = 2

skip_count = 0

def step4and5(cursor, dayoftheweek, hour, short_date, avg_co):
    """ Implement Step 4 and Step 5 of the plan - get all the average co values, filtering out the specific co value"""
    #Step 4 - Get all the average_co from the fixed table filtering get the specific co value
    # sort by closest value to avg_co, then by closest date to the query date
    sql_str = """select datetime, weekdays, dayoftheweek ,co, avg_co from FixedSamples where dayoftheweek="{0}" and time="{1}"and date!="{2}" group by datetime order by ABS(avg_co-{3}) asc, ABS(datediff("{2}",date)) asc;""".format(dayoftheweek, hour, short_date, avg_co)
    # start date
    cursor.execute(sql_str)
    results = cursor.fetchall()
    
    #Step 5 - find the 2 closest values to Step 3
    if len(results) == 0:
        n_closest = [None for x in xrange(N)]
    elif len(results) == 1:
        n_closest = [results[x] for x in xrange(len(results))]
        n_closest += [None ]
        #n_closest = [results[x] for x in xrange(N)]
    else:
        n_closest = [results[x] for x in xrange(N)]
    #for x in xrange(n):
    #    print dayoftheweek, n_closest[x]
    #return closest values, excluding query date
    return (True, n_closest)

def get_estimates_value(cursor, day_to_closest, location):
    """ Get the estimates values to be put into the model """
    # Step 7 - get array of values based on weekday or weekend
    global skip_count
    estimate_values = {} 
    for k, val in day_to_closest.items():
        vals = []
        for row in val:
            try:
                sql_str = """select co from Estimates where grid_location={0} and datetime="{1}";""".format(location, row[0])
                cursor.execute(sql_str)
                result = cursor.fetchall()
                #print row[0], result, sql_str
                #print result
                if len(result) == 0:
                    vals.append([0.2])
                else:
                    vals.append(result[0])
            except TypeError:
                if row is None:
                    vals.append([0.2])
                    continue
                else:
                    print "Problems with the script"
                    sys.exit()
        estimate_values[k] = vals
    #print estimate_values
    return (True, estimate_values)
    
def populate_model_table(cursor, input_datetime, grid_location, day_to_co, co):
    """ Populate the Model table """
    # Step 8 - populate the model table
    # if any of the values are less than 5, it's a weekday
    if any([x < 5 for x in day_to_co.keys()]):
        table = "WeekdayModelData"
    else:
        table = "WeekendModelData"

    #for k, v in day_to_co.items():
    # populate the right sql model table
    data = ["'{0}'".format(input_datetime), "{0}".format(grid_location)]
    feature_data = list(np.array([[day_to_co[k][i] for i in xrange(len(day_to_co[k]))] for k in sorted(day_to_co.keys())]).flatten())
    feature_data = ["{0}".format(x) for x in feature_data]
    data += feature_data
    data += ["{0}".format(co)]
    #print data
    # TODO should check the ordering so that the data per day is correctly inserted into the table in the correct order
    insert_str = """insert ignore into {0} () values ({1}); """.format(table, ','.join(data))
    #print insert_str
    cursor.execute(insert_str)

def main():
    #Step 3 - Get the output from the fixed table
    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    # input date and location TODO - how are we going to do this?
    #input_datetime = "2015-09-03 10:00:00"
    #lat = -31.34545
    #lon = 152.34545
    #location = (lat, lon)
    #location = 6500
    #co = 1.2345

    #print "input datetime: ", input_datetime
    #datetime_obj = datetime.strptime(input_datetime, "%Y-%m-%d %H:%M:%S") 

    # get the oldest date
    sql_str = """select distinct date from Samples  where user_id = 2 order by date asc limit 1;"""
    # start date
    cursor.execute(sql_str)
    start_date = cursor.fetchone()[0]
    #override TODO
    start_date = datetime(2015,8,1)

    # get the newest date
    sql_str = """select distinct date from Samples  where user_id = 2 order by date desc limit 1;"""
    # end date
    cursor.execute(sql_str)
    end_date = cursor.fetchone()[0]
    #override TODO
    end_date = datetime(2015,10,1)

    #provide some buffer time (extra epoch)
    epochs =  ((end_date - start_date).days*24 + 1)
                
    print "Start data and end date: {0} to {1}".format(start_date, end_date)
    print "Number of hours of data: {0}".format(epochs)

    first_date = start_date

    row_count = 0

    for _ in xrange(epochs):
        #get the day of the week
        dayoftheweek = first_date.weekday()
        
        if dayoftheweek < 5:
            table = "WeekdayModelData"
        else:
            table = "WeekendModelData"

        #check if a datetime is already in the db
        sql_str = """select count(*) from {0} where datetime="{1}";""".format(table, first_date)
        cursor.execute(sql_str)
        results = cursor.fetchone()

        # skip to next hour if there's something in there
        if results[0] > 0:
            print "Skipped entry for ", first_date
            first_date += timedelta(seconds=3600)
            continue
        
        #get all the rows for estimates for an hour (10,000)
        sql_str = """select datetime, grid_location, co from Estimates where datetime="{0}" order by grid_location asc;""".format(first_date)
        cursor.execute(sql_str)
        results = cursor.fetchall()

        short_date = first_date.date().strftime("%Y-%m-%d")
        for row in results:
            # get the location variables
            location = row[1]
            hour = first_date.hour
            co = row[2]

            # Step 3 - get the specific co value
            sql_str = """select avg_co, datetime from FixedSamples where date="{0}" and time="{1}" order by date asc limit 1;""".format(short_date, hour)
            #print sql_str
            # start date
            cursor.execute(sql_str)
            result = cursor.fetchall()
            if len(result) == 0:
                break;
            result = result[0]
            avg_co = result[0]
            #print avg_co, result[1]
            result_datetime = result[1]
            #print result_datetime, first_date
            assert result_datetime.strftime("%Y-%m-%d %H:%M:%S") == first_date.strftime("%Y-%m-%d %H:%M:%S")
            #print "avg_co is ", avg_co
            
            #Step 4 and 5 - map day to closest records in FixedSamples
            day_to_closest = {}
            (valid, day_to_closest[dayoftheweek]) = step4and5(cursor, dayoftheweek, hour, short_date, avg_co)

            # Step 6 - Find the rest of the days in the week
            # if it's a weekday
            if dayoftheweek < 5:
                days_left = set(range(5))
            # it's a weekend
            else:
                days_left = set(range(5,7))

            #remove the current day
            days_left.remove(dayoftheweek)
            # for the rest of the days find the closest avg_co values
            for day in days_left:
                #Repeat step 4 and 5
                (valid,day_to_closest[day]) = step4and5(cursor, day, hour, short_date, avg_co)

            #pprint for debugging
            #pp.pprint(day_to_closest)
            
            #Step 7 - Get CO array given the date, time and the location input
            (valid, day_to_co) = get_estimates_value(cursor, day_to_closest, location)
            
            #pprint for debugging
            #pp.pprint(day_to_co)

            #populate the respect table needed for training the model
            populate_model_table(cursor, first_date, location, day_to_co, co)
            row_count += 1
       
        # commit after every hour (10,000 rows)
        db.commit()
        
        #print
        print "Row total inserted is ", row_count, "populated at ", first_date, "Skip total", skip_count
        first_date += timedelta(seconds=3600)

    # close the connection to the db
    db.close()



if __name__ == "__main__":
    print "Starting script"
    # execute only if run as a script
    main()
    print "Script finished!"

