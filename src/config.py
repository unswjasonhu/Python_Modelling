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

class TestingConfig(Config):
    TESTING = True

