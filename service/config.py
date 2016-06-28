import os

RECOMMENDER_SECRET_KEY = 'this should be changed'
RECOMMENDER_MAX_HITS = 10000
RECOMMENDER_MAX_INPUT = 500
RECOMMENDER_CHUNK_SIZE = 100
RECOMMENDER_MAX_NEIGHBORS = 40
RECOMMENDER_NUMBER_SUGGESTIONS = 10
RECOMMENDER_THRESHOLD_FREQUENCY = 1
RECOMMENDER_SOLR_PATH = 'https://api.adsabs.harvard.edu/v1/search/query'
RECOMMENDER_CLUSTER_PROJECTION_PATH = 'data/clusters'
SQLALCHEMY_BINDS = {
    'recommender': 'postgresql+psycopg2://user:pwd@localhost:5432/recommender'}
# In what environment are we?
ENVIRONMENT = os.getenv('ENVIRONMENT', 'staging').lower()
# Config for logging
RECOMMENDER_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(levelname)s\t%(process)d '
                      '[%(asctime)s]:\t%(message)s',
            'datefmt': '%m/%d/%Y %H:%M:%S',
        }
    },
    'handlers': {
        'file': {
            'formatter': 'default',
            'level': 'INFO',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': '/tmp/recommender_service.app.{}.log'.format(ENVIRONMENT),
        },
        'console': {
            'formatter': 'default',
            'level': 'INFO',
            'class': 'logging.StreamHandler'
        },
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
# Define the autodiscovery endpoint
DISCOVERER_PUBLISH_ENDPOINT = '/resources'
# Advertise its own route within DISCOVERER_PUBLISH_ENDPOINT
DISCOVERER_SELF_PUBLISH = False
