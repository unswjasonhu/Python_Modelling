#! /usr/bin/python

import MySQLdb
import sys
from datetime import timedelta, datetime

from ..resources import get_flattened_index


def main():
    # Open database connection
    db = MySQLdb.connect("localhost","pollution","pollution","pollution_monitoring" )

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

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
    # override TODO
    end_date = datetime(2015,10,1)

    epochs =  ((end_date - start_date).days/7 +1)
    print "Number of weeks of data: {0}".format(epochs)
    #filtered out lat/lon null and co is null
    first_date = start_date
    # for each week
    total_rows = 0
    for _ in xrange(epochs):
        #get the aggregated values for use later
        agg_str = """select date, avg(co) from Samples where user_id = 2 and (location_name="CHULLORA" or location_name="PROSPECT" or location_name="ROZELLE" or location_name="LIVERPOOL") and date between "{0}" and date_add("{0}", interval 7 day) and co is not null and latitude is not null and longitude is not null group by date order by date asc;""".format(first_date)
        
        cursor.execute(agg_str)
        agg_results = cursor.fetchall()
        agg_results = dict(agg_results)
        # get data
        select_str = """select date, DATE_FORMAT(date,"%Y-%m-%d"), DATE_FORMAT(date,"%H"), if(WEEKDAY(date)<5, true, false), WEEKDAY(date), latitude, longitude, location_name, co from Samples where user_id = 2 and (location_name="CHULLORA" or location_name="PROSPECT" or location_name="ROZELLE" or location_name="LIVERPOOL") and date between "{0}" and date_add("{0}", interval 7 day) and co is not null and latitude is not null and longitude is not null order by date asc """.format(first_date)
        cursor.execute(select_str)
        results = cursor.fetchall()
        total_rows += len(results)
        print "Number of rows of Samples considered in total: {0}".format(total_rows)
        for result in results:
            #check if the date has associated sensor data
            input_datetime = result[0]
            select_str = """select * from Samples where user_id != 2 and date like "{0}%" and co < 60 and co > 0 limit 1;""".format(input_datetime.strftime("%Y-%m-%d %H"))
            cursor.execute(select_str)
            if cursor.rowcount == 0:
                print "Skipped {0} due to lack of sensor data".format(input_datetime)
                continue;
            agg_for_date = agg_results[input_datetime]
            data = ['"{0}"'.format(result[x]) for x in xrange(2)] + [result[x] for x in xrange(2,5)] + [get_flattened_index(float(result[5]), float(result[6]))] + ['"{0}"'.format(result[7])] + [result[8] , agg_for_date]
            data = ['{0}'.format(x) for x in data]
            # input data into sql
            insert_str = """insert ignore into FixedSamples () values ({0}); """.format(','.join(data))
            cursor.execute(insert_str)
        # commit
        db.commit()
        first_date = first_date + timedelta(days=7) 
    db.close()
        
if __name__ == "__main__":
        print "Starting script"
        # execute only if run as a script
        main()
        print "Script finished!"
