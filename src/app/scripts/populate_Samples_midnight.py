#! /usr/bin/python

import MySQLdb
import sys
from datetime import timedelta

import pprint

pp = pprint.PrettyPrinter(indent=4)

cmd_folder = '/code'
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
from src.config import config



def main():
    """
    This script is a fix to populate hours where there is no midnight co value.
    The script essentially fixes a bug with the Samples table population.
    """
    # Open database connection
    db = MySQLdb.connect(config.DATABASE_URI, config.DATABASE_USER, config.DATABASE_PASSWORD, config.DATABASE_NAME)

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    # get the oldest date
    sql_str = """select distinct date from Samples  where user_id = 2 order by date asc limit 1;"""

    # start date
    cursor.execute(sql_str)
    start_date = cursor.fetchone()[0]
    #override TODO
    #start_date = datetime(2015,8,1)


    # get the newest date
    sql_str = """select distinct date from Samples  where user_id = 2 order by date desc limit 1;"""

    # end date
    cursor.execute(sql_str)
    end_date = cursor.fetchone()[0]
    # override TODO
    #end_date = datetime(2015,10,1)

    # get the newest date
    sql_str = """select DATE_FORMAT(date,"%Y-%m-%d") as shortdate, Samples.* from Samples where user_id = 2 and (location_name="CHULLORA" or location_name="PROSPECT" or location_name="ROZELLE" or location_name="LIVERPOOL") and date between "{0}" and "{1}" and (DATE_FORMAT(date,"%H") = "00" or DATE_FORMAT(date,"%H") = "01") and co is not NULL  order by date desc;""".format(start_date, end_date)
    short_date_index = 0
    location_index = 7
    print(sql_str)

    #column information for the table, not the query
    # date, latitude, longitude latitude   | longitude  | location_error | computed_location | location_name   | user_id | group_id | device_id | temperature | humidity | speed | co2  | co
    columns = ["date","latitude",  "longitude", "location_error", "computed_location", "location_name", "user_id", "group_id", "device_id", "temperature", "humidity", "speed", "co2", "co"]
    columns_str = ','.join(columns)
    # end date
    cursor.execute(sql_str)
    results = cursor.fetchall()

    #2d dict - 1st dim is short_date, 2nd dim is location
    results_dict = {}
    for result in results:
        #if the short date isn't there
        if result[short_date_index] not in results_dict.keys():
            results_dict[result[short_date_index]] = {}
            upper_location = result[location_index].upper()
            results_dict[result[short_date_index]][upper_location] = [result]
        else:
            #if the location is not in the dict
            upper_location = result[location_index].upper()
            if upper_location not in results_dict[result[short_date_index]].keys():
                results_dict[result[short_date_index]][upper_location] = [result]
            else:
                results_dict[result[short_date_index]][upper_location].append(result)

    def stringify(x):
        """ convert specified types to string"""
        return str(x)

    assert "2010-04-05" in results_dict.keys()


    # for each week
    total_rows = 0
    skip_rows = 0

    for date, rows  in results_dict.items():
        #if date == "2010-04-04":
        #    print len(rows)
        #    print date, rows
        #    print results_dict[date]
        #    sys.exit()
        for location, rows in results_dict[date].items():
            #gather the dates from the rows
            data = None
            midnight_exists = False
            for row in list(rows):
                if row[2].hour == 1:
                    #data is used later
                    data = list(row)
                if row[2].hour == 0:
                    #check whether the location exists
                    sql_str = """select date, location_name from Samples where user_id = 2 and location_name="{0}" and date="{1} 00:00:00";""".format(location, date)
                    cursor.execute(sql_str)
                    results = cursor.fetchall()
                    #does the row exist already with the locationa nd date
                    if len(results) > 0 :
                        midnight_exists = True
                        #skip because the row already exists
                        skip_rows += 1
                        if skip_rows %50 == 0:
                            print("Skipped rows ", skip_rows)
                        break;
            #go to the next location, but same date if the midnight value exists
            if midnight_exists:
                continue;

            #############
            #continue if no midnight value exists
            #use data for input data
            #data = list(list(rows)[0])
            #the case below is no 01:00:00 exists for the particular day, different to if a NULL exists for a co
            if data is None:
                print("No 01:00:00 data for {0} at {1}".format(date,location))
                sys.exit()

            # replace None with ''
            for i, x in enumerate(data):
                if x is None:
                    data[i] = 'NULL'

            #data to input into Samples
            input_data = []
            #new date - should be 00:00:00
            input_data.append((data[2] - timedelta(hours=1)).strftime('"%Y-%m-%d %H:%M:%S"'))
            #insert first part up until computed location
            input_data += [stringify(x) for x in data[3:7]]
            # location name
            input_data += ['"{0}"'.format(x) for x in data[7:8]]
            print(['"{0}"'.format(x) for x in data[7:8]])
            #store user id and group id
            input_data += ["{0}".format(x) for x in data[8:10]]
            #store the rest of the columns
            input_data += [stringify(x) for x in data[10:15]]
            input_data += [stringify(data[15])]
            #print input_data
            #print columns

            assert len(input_data) == len(columns)
            #print data[10:16]
            #print input_data
            insert_str = """insert ignore into Samples ({0}) values ({1}); """.format(columns_str, ','.join(input_data))
            #print insert_str
            cursor.execute(insert_str)
            total_rows += 1
            #print results_dict.values()
            #sys.exit()

        # commit
        db.commit()
        if total_rows %50 == 0 or skip_rows %50 ==0:
            print("Rows inserted so far ", total_rows, ", Skip rows ", skip_rows)

    print("Total rows inserted ", total_rows)
    print("Skipped rows ", skip_rows)

    db.close()

if __name__ == "__main__":
        print("Starting script")
        # execute only if run as a script
        main()
        print("Script finished!")
