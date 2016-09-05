/* Make the Samples run faster */
CREATE INDEX user_Samples ON Samples (user_id);

/* To store fixed Sample data */
DROP TABLE IF EXISTS FixedSamples;
CREATE TABLE FixedSamples
(
datetime datetime,
date date,
time int,
weekdays int,
dayoftheweek int,
grid_location int,
location_name varchar(64),
co decimal(10,5),
avg_co decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, location_name)
);
CREATE INDEX date_FixedSamples ON FixedSamples (date);

/* To store Estimates for hours in all grid locations. i.e. 10,000 rows per hour, for a 100x100 grid of Sydney */
DROP TABLE IF EXISTS Estimates; 
CREATE TABLE Estimates 
(
datetime datetime,
date date,
time int,
weekdays int,
dayoftheweek int,
grid_location int,
co decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location)
);
CREATE INDEX date_Estimates ON Estimates (date);

/* To store Estimates for hours in all grid locations, subtracted by the mean. i.e. 10,000 rows per hour, for a 100x100 grid of Sydney */
DROP TABLE IF EXISTS Estimates_zeroMean;
CREATE TABLE Estimates_zeroMean 
(
datetime datetime,
date date,
time int,
weekdays int,
dayoftheweek int,
season int,
grid_location int,
co decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location)
);
CREATE INDEX date_Estimates_zeroMean ON Estimates_zeroMean (date);

/* To store samples data */
DROP TABLE IF EXISTS samplesGridData;
CREATE TABLE samplesGridData
(
datetime datetime,
date date,
time int,
weekdays int,
dayoftheweek int,
season int,
grid_location_row int,
grid_location_col int,
co_chullora decimal(10,5),
co_liverpool decimal(10,5),
co_prospect decimal(10,5),
co_rozelle decimal(10,5),
co_original decimal(10,5),
co decimal(10,5),
co_mean decimal(10,5),
co_stddev decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location_row, grid_location_col)
);
CREATE INDEX date_samplesGridData ON samplesGridData (date);

/* To store SVM estimates data */
DROP TABLE IF EXISTS SVMEstimates;
CREATE TABLE SVMEstimates
(
datetime datetime,
date date,
time int,
weekdays int,
dayoftheweek int,
season int,
grid_location_row int,
grid_location_col int,
co_chullora decimal(10,5),
co_liverpool decimal(10,5),
co_prospect decimal(10,5),
co_rozelle decimal(10,5),
co_original decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location_row, grid_location_col)
);
CREATE INDEX date_SVMEstimates ON SVMEstimates (date);

/* To store sampoleGridData data with minute granularity */
DROP TABLE IF EXISTS samplesGridDataMinutes;
CREATE TABLE samplesGridDataMinutes
(
datetime datetime,
date date,
time int,
minute int,
weekdays int,
dayoftheweek int,
season int,
grid_location_row int,
grid_location_col int,
co_chullora decimal(10,5),
co_liverpool decimal(10,5),
co_prospect decimal(10,5),
co_rozelle decimal(10,5),
co_original decimal(10,5),
co decimal(10,5),
co_mean decimal(10,5),
co_stddev decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location_row, grid_location_col)
);
CREATE INDEX date_samplesGridDataMinutes ON samplesGridDataMinutes (date);

/* To store mean and std values for the sensor network, and mean values for the fixed station. This is for part 1 and 2 */
DROP TABLE IF EXISTS CV_values;
CREATE TABLE CV_values
(
datetime datetime,
date date,
time int,
weekdays int,
dayoftheweek int,
mean_sensor decimal(10,5),
std_sensor decimal(10,5),
mean_fixed decimal(10,5),
grids int,
CONSTRAINT pk PRIMARY KEY (datetime)
);

CREATE INDEX date_CV_values ON CV_values (date);



/* Data to feed into the Weekday Model */
DROP TABLE IF EXISTS WeekdayModelData; 
CREATE TABLE WeekdayModelData
(
datetime datetime,
grid_location int,
co1 decimal(10,5),
co2 decimal(10,5),
co3 decimal(10,5),
co4 decimal(10,5),
co5 decimal(10,5),
co6 decimal(10,5),
co7 decimal(10,5),
co8 decimal(10,5),
co9 decimal(10,5),
co10 decimal(10,5),
co decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location)
);


/* Data to feed into the Weekend Model */
DROP TABLE IF EXISTS WeekendModelData; 
CREATE TABLE WeekendModelData
(
datetime datetime,
grid_location int,
co1 decimal(10,5),
co2 decimal(10,5),
co3 decimal(10,5),
co4 decimal(10,5),
co decimal(10,5),
CONSTRAINT pk PRIMARY KEY (datetime, grid_location)
);

/*
Original Field tables
-------------------+---------------+------+-----+----------+----------------+
| Field             | Type          | Null | Key | Default  | Extra          |
+-------------------+---------------+------+-----+----------+----------------+
| id                | int(11)       | NO   | PRI | NULL     | auto_increment |
| date              | datetime      | YES  | MUL | NULL     |                |
| latitude          | decimal(9,6)  | YES  |     | NULL     |                |
| longitude         | decimal(9,6)  | YES  |     | NULL     |                |
| location_error    | decimal(9,6)  | NO   |     | 0.000000 |                |
| computed_location | tinyint(1)    | NO   |     | 0        |                |
| location_name     | varchar(64)   | YES  |     | NULL     |                |
| user_id           | int(11)       | NO   |     | 0        |                |
| group_id          | int(11)       | NO   |     | 0        |                |
| device_id         | int(11)       | YES  |     | NULL     |                |
| temperature       | decimal(3,1)  | YES  |     | NULL     |                |
| humidity          | decimal(5,2)  | YES  |     | NULL     |                |
| speed             | decimal(5,2)  | YES  |     | NULL     |                |
| co2               | decimal(10,5) | YES  |     | NULL     |                |
| co                | decimal(10,5) | YES  |     | NULL     |                |
| pm10              | decimal(8,4)  | YES  |     | NULL     |                |
| pm2.5             | decimal(8,4)  | YES  |     | NULL     |                |
| no                | decimal(10,5) | YES  |     | NULL     |                |
| no2               | decimal(10,5) | YES  |     | NULL     |                |
| so2               | decimal(10,5) | YES  |     | NULL     |                |
| o3                | decimal(10,5) | YES  |     | NULL     |                |
+-------------------+---------------+------+-----+----------+----------------+
*/

