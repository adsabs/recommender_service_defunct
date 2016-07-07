import os

RECOMMENDER_SECRET_KEY = 'this should be changed'
# Maximum number of rows for Solr query
RECOMMENDER_MAX_HITS = 10000
# The maximum number of closest papers in a cluster to use in calculations
RECOMMENDER_MAX_NEIGHBORS = 40
# How many recommendations are generated
RECOMMENDER_NUMBER_SUGGESTIONS = 10
# Specify the endpoint for Solr queries
RECOMMENDER_SOLR_PATH = 'https://api.adsabs.harvard.edu/v1/search/query'
# Where are the data files used for cluster calculations
RECOMMENDER_CLUSTER_PROJECTION_PATH = 'data/clusters'
# Recommendations for publications before this year are very likely to be meaningless
RECOMMENDER_FROM_YEAR = 1997
# Recommendations for other journals than these are probably meaningless
RECOMMENDER_ALLOWED_JOURNALS = ['ApJ..','ApJS.','AJ...','MNRAS','A&A..','ARA&A','PhRvD','A&ARv','PASP.','PASJ.','A&C..']
# Specify where the recommender database lives
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
