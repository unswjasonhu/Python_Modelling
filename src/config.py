import os

class Config(object):
    DEBUG = False
    TESTING = False
    DATABASE_NAME='pollution_monitoring'
    DATABASE_USER='pollution'
    DATABASE_PASSWORD='pollution'

class ProductionConfig(Config):
    DATABASE_URI = 'localhost'
    ENVIRONMENT='prod'

class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URI = 'db'
    ENVIRONMENT='dev'

class TestConfig(Config):
    TESTING = True
    DATABASE_NAME='test_pollution_monitoring'


def get_config(environment='prod'):
    if not environment:
        environment = os.environ['ENVIRONMENT']

    if environment == 'dev':
        return DevelopmentConfig()
    elif environment == 'prod':
        return ProductionConfig()
    elif environment == 'test':
        return TestConfig()

config = get_config()