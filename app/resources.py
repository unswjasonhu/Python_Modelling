#! /usr/bin/python

from __future__ import division
import sys
from math import radians, sin, cos, sqrt, asin, atan2, degrees
import time
from datetime import datetime
import MySQLdb
import pandas as pd
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates
from scipy.interpolate import griddata
from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

from invdisttree import Invdisttree

np.set_printoptions(threshold='nan')

IMAGES_BASE_DIR = "../images"
DEFAULT_HEATMAP_NAME = IMAGES_BASE_DIR + "/before_interpolation_averages"

# defining sensors
STATIONARY = [2]
NODE_OLD = [6]

# in the last two years
NODE_NEW = list(range(7, 12))

# all node sensor data
NODE = NODE_OLD + NODE_NEW

# all sensors
SENSORS = NODE + STATIONARY

data_sanity = 50

[NW_BOUND, SW_BOUND, NE_BOUND, SE_BOUND] = [(-33.728545, 150.849275),
                                            (-33.982863, 150.849275),
                                            (-33.728545, 151.24753),
                                            (-33.982863, 151.24753)]

# define grid square size
GRID_RES = 100


def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8  # Earth radius in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2)**2 + cos(lat1) * cos(lat2) * sin(dLon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

syd_grid_left_dist = haversine(NW_BOUND[0], NW_BOUND[1], SW_BOUND[0], SW_BOUND[1])
syd_grid_top_dist = haversine(NW_BOUND[0], NW_BOUND[1], NE_BOUND[0], NE_BOUND[1])

syd_grid_right_dist = haversine(NE_BOUND[0], NE_BOUND[1], SE_BOUND[0], SE_BOUND[1])
syd_grid_bottom_dist = haversine(SW_BOUND[0], SW_BOUND[1], SE_BOUND[0], SE_BOUND[1])

# assume that we don't care about the curvature of the earth,
# so top distance is the same as the bottom
# they're the same to the nearest kilometer
syd_grid_lon_dist = syd_grid_top_dist
syd_grid_lat_dist = syd_grid_left_dist


def classify_hour(hour):
    """ Classify the hour based on some aggregate ranges"""
    if hour >= 8 and hour <= 11:
        return 1
    elif hour >= 12 and hour <= 15:
        return 2
    elif hour >= 16 and hour <= 19:
        return 3
    elif hour >= 20 and hour <= 23:
        return 4
    raise Exception("Hour out of bounds")


def get_season(user_datetime):
    """ Gets the seasons, summer=0, autumn=1, etc """
    user_month = user_datetime.month
    if user_month == 12 or user_month <= 2:
        return 0
    elif user_month >= 3 and user_month <= 5:
        return 1
    elif user_month >= 6 and user_month <= 8:
        return 2
    elif user_month >= 9 and user_month <= 11:
        return 3


def find_coords_given_bearing(lat1, lon1, bearing, distance):
    """ Find the coords given a start point, distance and bearing"""
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    bearing = radians(bearing)
    R = 6372.8  # Earth radius in kilometers
    lat2 = asin(sin(lat1) * cos(distance / R) + cos(lat1) * sin(distance / R) * cos(bearing))
    lon2 = lon1 + atan2(sin(bearing) * sin(distance / R) * cos(lat1),
                        cos(distance / R) - sin(lat1) * sin(lat2))
    return (degrees(lat2), degrees(lon2))


def check_stagnant(grouped):
    """ checks if the coords given contain many grid locations """
    locations = set()
    for coord, _ in grouped:
        locations.add(get_flattened_index(round(coord[0], 6), round(coord[1], 6)))
    if len(locations) == 1:
        return True
    return False


def get_index(latitude, longitude):
    """ Get the grid index in a region covering Sydney """
    lat_index = haversine(NW_BOUND[0], NW_BOUND[1], latitude, NW_BOUND[
                          1]) / syd_grid_lat_dist * (GRID_RES - 1)
    lon_index = haversine(NW_BOUND[0], NW_BOUND[1], NW_BOUND[0],
                          longitude) / syd_grid_lon_dist * (GRID_RES - 1)
    return int(lat_index), int(lon_index)


def get_flattened_index(latitude, longitude):
    """ Get the grid index in a grid flattened to a 1d array, for a region covering Sydney """
    lat_index = haversine(NW_BOUND[0], NW_BOUND[1], latitude, NW_BOUND[
                          1]) / syd_grid_lat_dist * (GRID_RES - 1)
    lon_index = haversine(NW_BOUND[0], NW_BOUND[1], NW_BOUND[0],
                          longitude) / syd_grid_lon_dist * (GRID_RES - 1)
    # print NW_BOUND[0],NW_BOUND[1],NW_BOUND[0],longitude
    return (int(lat_index) * GRID_RES) + int(lon_index)


def get_coords_sydney(centre_offset=False):
    """ Return a 100x100 array with the coords of sydney """
    coords = [[0 for i in range(GRID_RES)] for x in range(GRID_RES)]
    lat = NW_BOUND[0]
    lon = NW_BOUND[1]
    start_lon = NW_BOUND[1]

    if centre_offset:
        lon = find_coords_given_bearing(lat, lon, 90, syd_grid_lon_dist / GRID_RES / 2)[1]
        lat = find_coords_given_bearing(lat, lon, 180, syd_grid_lat_dist / GRID_RES / 2)[0]

    for i in range(GRID_RES):
        for j in range(GRID_RES):
            lon = find_coords_given_bearing(lat, lon, 90, syd_grid_lon_dist / GRID_RES)[1]
            coords[i][j] = (lat, lon)
        lon = start_lon
        lat = find_coords_given_bearing(lat, lon, 180, syd_grid_lat_dist / GRID_RES)[0]

    return coords


def create_mesh(grid, name, title_name='interpolated', plane=True):
    """ Create mesh for a given grid, and a plane for the mean"""
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y = range(GRID_RES), range(GRID_RES)
    x, y = np.meshgrid(x, y)

    intensity = np.array(grid)
    Zm = ma.array(intensity, mask=np.isnan(intensity))

    surf = ax.plot_surface(x, y, Zm, rstride=1, cstride=1,
                           cmap=cm.hot, linewidth=0, antialiased=False)

    ax.plot_surface(x, y, Zm.mean(), alpha=0.2, color='k')

    ax.set_zlim(intensity.min(), intensity.max())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Mesh for ' + title_name + ' results')
    fig.set_size_inches(18.5, 10.5)

    plt.savefig('{0}.png'.format(name), dpi=100)
    plt.clf()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    az = 0
    ax.view_init(0, az)
    surf = ax.plot_surface(x, y, Zm, rstride=1, cstride=1,
                           cmap=cm.hot, linewidth=0, antialiased=False)

    ax.plot_surface(x, y, Zm.mean(), alpha=0.2, color='k')

    ax.set_zlim(intensity.min(), intensity.max())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Mesh for ' + title_name + ', view elevation=0,azimuth=' + str(az))

    fig.set_size_inches(18.5, 10.5)
    plt.savefig('{0}_alt.png'.format(name), dpi=100)
    plt.close()


def create_heatmap(grid, name, strip=False):
    """ Create heatmap given a grid, save it with the name provided"""

    # setup the 2D grid with Numpy
    x, y = range(GRID_RES), range(GRID_RES)
    xx, yy = np.meshgrid(x, y)

    # convert intensity (list of lists) to a numpy array for plotting
    intensity = np.array(grid)
    Zm = ma.array(intensity, mask=np.isnan(intensity))

    # now just plug the data into pcolormesh, it's that easy!
    if strip:
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        # set the colormap
        jet = plt.get_cmap('jet')
        plt.pcolormesh(xx, yy, Zm[::-1], vmin=0, vmax=10, cmap=jet)
        fig.canvas.print_png('{0}.png'.format(name), bbox_inches='tight')
    else:
        plt.pcolormesh(xx, yy, Zm[::-1])
        # need a colorbar to show the intensity scale
        plt.colorbar()
        plt.savefig('{0}.png'.format(name))
    plt.close()


def idw_interpol(known, z, ask, Nnear=8):
    """ Interpolate using IDW algorithm """
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 1  # weights ~ 1 / distance**p

    invdisttree = Invdisttree(known, z, leafsize=leafsize, stat=1)
    interpol = invdisttree(ask, nnear=Nnear, eps=eps, p=p)
    interpol = interpol.reshape(GRID_RES, GRID_RES)
    return (interpol, "idw")


def griddata_interpol(known, z, ask):
    """ Interpolate using scipy's griddata """
    grid_x, grid_y = np.mgrid[0:GRID_RES:1, 0:GRID_RES:1]
    interpol = griddata(known, z, ask, method='cubic')
    return (interpol, "griddata_cubic")


def create_mean_value_grid(df):
    """ Bin all the data in the dataframe for a region bounded by Sydney.
    Creates 100x100 matrix where each element is a list of values"""

    # create an empty grid, with each element as a list
    expanded_grid = [[[] for x in range(GRID_RES)] for x in range(GRID_RES)]
    # need a way to map bounds to a number between 0 and 99
    row_iterator = df.iterrows()

    # for each row in the data frame
    for i, row in row_iterator:
        lat_index, lon_index = get_index(row['latitude'], row['longitude'])
        expanded_grid[lat_index][lon_index].append(row['co'])

    # create zeros array
    sydney_grid = np.zeros((GRID_RES, GRID_RES))

    # do the aggregations and put it into a numpy array
    count = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            sydney_grid[i][j] = np.mean(expanded_grid[i][j])
            if not np.isnan(sydney_grid[i][j]):
                count += 1

    return sydney_grid, count


def gridify_sydney(df, heatmap_name=DEFAULT_HEATMAP_NAME, verbose=True, heatmap=True):
    """ Convert Sydney to a grid """
    sydney_grid, count = create_mean_value_grid(df)

    # do some aggregation function over the buckets to get the final grid
    # do the interpolation
    # get values for interpolation
    known = []
    z = []
    ask = []
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            ask.append([i, j])
            if not np.isnan(sydney_grid[i][j]):
                # known
                known.append([i, j])
                # z
                z.append(sydney_grid[i][j])
    known = np.array(known)
    z = np.array(z)
    ask = np.array(ask)

    sydney_grid = np.array(sydney_grid)
    if verbose:
        print("Before interpolation, after averaging: Min, max, mean: {0}, {1}, {2}".format(
            np.nanmin(sydney_grid), np.nanmax(sydney_grid), np.nanmean(sydney_grid)))
        print("Database values:{0}, fill rate:{1}".format(
            count, count / (GRID_RES * GRID_RES) * 100))
    if heatmap:
        create_heatmap(sydney_grid, heatmap_name)

    return known, z, ask, sydney_grid


def interpol_general(known, z, ask):
    """ Function which calls the specific interpolation method """
    # Interpolation
    (interpol, interpol_name) = idw_interpol(known, z, ask, len(z))

    print("Interpolation values: Min, max, mean and range: {0}, {1}, {2}, {3}".format(
        interpol.min(), interpol.max(), interpol.mean(), interpol.max() - interpol.min()))
    create_heatmap(interpol, IMAGES_BASE_DIR + "/after_" + interpol_name + "_interpolation")
    create_mesh(interpol, IMAGES_BASE_DIR + "/after_mesh_" + interpol_name + "_interpolation")

    return interpol


def create_histogram(df, name):
    """ Create a histgram for CO """
    df.hist(column=['co'])
    plt.savefig('{0}.png'.format(name))
    plt.close()


def data_from_db(sql_string, verbose=True, exit_on_zero=True):
    """ Retrieve data from the DB and put it into a pandas array """
    try:
        # Open database connection
        mysql_cn = MySQLdb.connect("localhost", "pollution", "pollution", "pollution_monitoring")
        df_mysql = pd.read_sql(sql_string, con=mysql_cn)
        if len(df_mysql.index) == 0:
            if exit_on_zero:
                print("No rows returned. Exiting...")
                sys.exit()
            else:
                return None
        if verbose:
            print("{0} rows returned".format(len(df_mysql.index)))
        return df_mysql
    except Exception as e:
        print(str(e))
        sys.exit()
    finally:
        mysql_cn.close()


def convert_to_localtime(series):
    """Convert real time series into local times"""
    series = matplotlib.dates.date2num([time.localtime(x) for x in series])
    print(series)
    return series


def model_NN(time_start, time_end, name="time_vs_co_averages"):
    """ Implement the NN model """
    name = IMAGES_BASE_DIR + "/" + name
    # retrieve sql data for the period required
    user_id = 7
    sql_string = """
        SELECT
            user_id,
            latitude,
            longitude,
            DATE_FORMAT(date, '%Y-%m-%d %H:%i') AS date, AVG(co) AS avg_co
        FROM
            Samples
        WHERE
            date BETWEEN '{0}' and '{1}' AND user_id = {2} AND co != 0
        GROUP BY
            DATE_FORMAT(date, '%Y-%m-%d %H:%i');
    """.format(time_start, time_end, user_id)

    df_mysql = data_from_db(sql_string)

    # plot the current data over the time period with a line and scatter to
    # get an idea of what it looks like
    df_mysql['date'] = pd.to_datetime(df_mysql['date'])
    plt.scatter(list(df_mysql["date"]), list(df_mysql["avg_co"]))
    plt.plot(list(df_mysql["date"]), list(df_mysql["avg_co"]))
    plt.xlim(df_mysql['date'].min(), df_mysql['date'].max())
    plt.ylim(df_mysql['avg_co'].min() - 0.1, df_mysql['avg_co'].max() + 0.1)
    plt.savefig('{0}.png'.format(name))
    plt.close()

    # Implement the NN model
    df_mysql = df_mysql.sort(['date'], ascending=1)
    co_matrix = df_mysql.as_matrix(columns=['date', 'avg_co'])
    X_train, y_train = co_matrix[:, 0], co_matrix[:, 1]

    # convert the dates to a decimal representation  so that they can be fit
    X_train = np.array([[matplotlib.dates.date2num(x)] for x in X_train])
    Z = np.float64(X_train)
    y = np.float64(y_train)

    X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=1 / 3, random_state=0)

    pipeline = Pipeline([
        ('min/max scaler', preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))),
        ('nn', Regressor(
            layers=[
                Layer("Rectifier", units=50),
                Layer("Linear")
            ],
            learning_rate=0.001,
            n_iter=70,
            valid_size=.1,
            verbose=True))
    ])

    optimise = False
    param_grid = {
        'nn__learning_rate': np.arange(0.001, 0.009, 0.001),
        'nn__hidden0__units': np.arange(25, 150, 25),
        'nn__hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]
    }

    gs = GridSearchCV(pipeline, param_grid=param_grid)

    if optimise:
        gs.fit(X_train, y_train)
        best_learning_rate = gs.best_params_['nn__learning_rate']
        best_hidden_layer_unit = gs.best_params_['nn__hidden0__units']
        best_hidden_layer_type = gs.best_params_['nn__hidden0__type']
        params = {
            'nn__learning_rate': best_learning_rate,
            'nn__hidden0__units': best_hidden_layer_unit,
            'nn__hidden0__type': best_hidden_layer_type
        }
        pipeline.set_params(**params)
        print(gs.best_params_)

    pipeline.fit(X_train, y_train)
    print(pipeline.get_params())
    y_pred = pipeline.predict(X_train)

    X_train = matplotlib.dates.num2date([x for x in X_train])
    plt.scatter(X_train, y_train)
    plt.plot_date(X_train, y_pred, fmt='go')
    plt.xlim(df_mysql['date'].min(), df_mysql['date'].max())
    plt.savefig('{0}.png'.format(name))
    plt.close()


def predict_with_model(model_file_location,
                       input_datetime=None,
                       input_date=None,
                       lat=None,
                       lon=None):
    if (not input_datetime) and (not input_date and not lat and not lon):
        # raise Exception('hello')
        return []
    # load current model
    fileObject = open(model_file_location, 'rb')
    pipeline = pickle.load(fileObject)

    # get data for model input
    sql_string = """
        SELECT
            date, location_name, if(WEEKDAY(date)<5, true, false) AS weekdays,
            WEEKDAY(date) AS dayoftheweek, co
        FROM
            Samples
        WHERE
            user_id = 2 AND
                date LIKE "{0}%"
                AND (location_name = "Prospect"
                    OR location_name = "Rozelle"
                    OR location_name = "Liverpool"
                    OR location_name = "Chullora")
        ORDER BY
            location_name;""".format(input_datetime or input_date)

    fixed_samples_data = data_from_db(sql_string, exit_on_zero=False)

    try:
        # assert that more than 4 stations need to be returned
        # sometimes 8 rows are returned (duplicate records..)
        assert fixed_samples_data is not None and len(fixed_samples_data) >= 4
    except AssertionError:
        return []

    try:

        FIXED_LOCATIONS = ['Chullora', 'Liverpool', 'Prospect', 'Rozelle']
        mean_fixed = np.nanmean([
            fixed_samples_data[fixed_samples_data.location_name == location]['co'].iloc[0] for location in FIXED_LOCATIONS
        ])
        co_chullora = fixed_samples_data[fixed_samples_data.location_name == 'Chullora']['co'].iloc[0] if not np.isnan(fixed_samples_data[fixed_samples_data.location_name == 'Chullora']['co'].iloc[0]) else mean_fixed
        co_liverpool = fixed_samples_data[fixed_samples_data.location_name == 'Liverpool']['co'].iloc[0] if not np.isnan(fixed_samples_data[fixed_samples_data.location_name == 'Liverpool']['co'].iloc[0]) else mean_fixed
        co_prospect = fixed_samples_data[fixed_samples_data.location_name == 'Prospect']['co'].iloc[0]  if not np.isnan(fixed_samples_data[fixed_samples_data.location_name == 'Prospect']['co'].iloc[0]) else mean_fixed
        co_rozelle = fixed_samples_data[fixed_samples_data.location_name == 'Rozelle']['co'].iloc[0]  if not np.isnan(fixed_samples_data[fixed_samples_data.location_name == 'Rozelle']['co'].iloc[0]) else mean_fixed
    except Exception:
        return []

    if input_datetime:
        input_datetime = datetime.strptime(input_datetime, "%Y-%m-%d %H:%M:%S")
        specific_hour = input_datetime.hour
        hour_feature = classify_hour(specific_hour)

        # Need the fixed station values
        data = [fixed_samples_data['weekdays'].iloc[0], hour_feature, get_season(
            input_datetime), 0, 0, co_liverpool, co_prospect, co_chullora, co_rozelle]
        X = np.float64([data])
        y_vals = []

        # go through 100x100 grid pixels
        for row in range(100):
            for col in range(100):
                X[0][3] = row
                X[0][4] = col
                y_vals.append((row, col, 5.7464 * pipeline.predict(X)[0] + 3.48652))

        return y_vals
    else:
        # Need the fixed station values
        input_date = datetime.strptime(input_date, "%Y-%m-%d")
        row, col = get_index(lat, lon)
        y_vals = []

        for hour in range(8, 24):
            hour_feature = classify_hour(hour)
            data = [fixed_samples_data['weekdays'].iloc[0], hour_feature, get_season(
                input_date), row, col, co_liverpool, co_prospect, co_chullora, co_rozelle]
            X = np.float64([data])
            y_vals.append((hour, 5.7464 * pipeline.predict(X)[0] + 3.48652))

        return y_vals


########################################################################
# Main starts here!
########################################################################

if __name__ == "__main__":
    test = False

    gridify = False
    NN = False

    if test:
        gridify = True
        NN = True

    if gridify:
        if len(sys.argv) == 2:
            date_test_str = sys.argv[1]
        else:
            date_test_str = "2015-09-03"

        print("date used is: " + date_test_str)

        ############
        # Get the spatial data for this date
        ############

        # ignore rows with no lat/lon info, and where co is NULL
        sql_string = """
            SELECT
                date,
                latitude,
                longitude,
                location_error,
                computed_location,
                location_name,
                user_id,
                group_id,
                device_id,
                co
            FROM
                Samples
            WHERE
                date LIKE "{0}%"
                    AND (latitude IS NOT NULL AND longitude IS NOT NULL)
                    AND (latitude <= {1} AND latitude >= {2})
                    AND (longitude >= {3} AND longitude <= {4})
                    AND co IS NOT NULL AND co <= 50;
        """.format(date_test_str, NW_BOUND[0], SW_BOUND[0], NW_BOUND[1], NE_BOUND[1])

        df_mysql = data_from_db(sql_string)

        co_col = df_mysql["co"]

        print("Data from DB: Min, max and range: {0}, {1}, {2}".format(
            co_col.min(), co_col.max(), co_col.max() - co_col.min()))

        # create a grid of sydney
        known, z, ask, sydney_grid = gridify_sydney(df_mysql)

        # TODO - print a series of images over 1 minute intervals
        # minutify_data(df_mysql)

        # do the interpolation
        interpolation_grid = interpol_general(known, z, ask)

    ############
    # Model the NN for a particular day and time, test time is going to be 7:30AM to 10:00AM.
    # First take a look at the averages over the minute and
    # try to establish where this person moved over time.
    ############
    if NN:
        # hard code the date_times
        print("Modelling using NN")
        date_time_start = "2015-09-03 10:30"
        date_time_end = "2015-09-03 11:00"
        name = "averages_" + date_time_start + "_" + date_time_end
        model_NN(date_time_start, date_time_end, name)
