##import api as fitbit
import fitbit
import ConfigParser

consumer_key = '2c10aacf7523ee1b71aa94099ade81cd'
consumer_secret = '4755dd9e99f93f77b23d71967d93bb71'

# unauth_client = fitbit.Fitbit(consumer_key, consumer_secret)
# certain methods do not require user keys
# print unauth_client.food_units()

# You'll have to gather the user keys on your own, or try
# ./gather_keys_cli.py <consumer_key> <consumer_secret> for development
# authd_client = fitbit.Fitbit(consumer_key, consumer_secret, resource_owner_key='<user_key>', resource_owner_secret='<user_secret>')
# authd_client = fitbit.Fitbit(consumer_key, consumer_secret)
# token = authd_client.fetch_request_token()

# for my fitbit account specifically
# gather_keys_cli.py 2c10aacf7523ee1b71aa94099ade81cd 4755dd9e99f93f77b23d71967d93bb71
##{   u'encoded_user_id': u'3RZRR6',
##    u'oauth_token': u'537d3981665952d113a76b17d179d692',
##    u'oauth_token_secret': u'720f3b1efc7b9c6fed50177738cef559'}
##RESPONSE
##{   u'encoded_user_id': u'3RZRR6',
##    u'oauth_token': u'537d3981665952d113a76b17d179d692',
##    u'oauth_token_secret': u'720f3b1efc7b9c6fed50177738cef559'}

user_key = '537d3981665952d113a76b17d179d692'
user_secret = '720f3b1efc7b9c6fed50177738cef559'

# print user_key

authd_client = fitbit.Fitbit(consumer_key, consumer_secret, oauth2=False, resource_owner_key=user_key, resource_owner_secret=user_secret)

#fitbit_stats = authd_client.intraday_time_series("activities/steps",base_date='today', detail_level='1min', start_time=None, end_time=None)
#print authd_client
fitbit_stats = authd_client.time_series("activities/steps", base_date='today', period='1d', end_date=None)

for key in fitbit_stats.keys():
    print(key, fitbit_stats[key])

#print authd_client.sleep()
