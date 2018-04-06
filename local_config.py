import logging
LOG_LEVEL = 30 # To be deprecated when all microservices use ADSFlask
LOGGING_LEVEL = "INFO"
RECOMMENDER_API_TOKEN = ""
RECOMMENDER_CHUNK_SIZE = 1
RECOMMENDER_CLUSTER_PROJECTION_PATH = ""
RECOMMENDER_MAX_HITS = 1
RECOMMENDER_MAX_INPUT = 1
RECOMMENDER_MAX_NEIGHBORS = 1
RECOMMENDER_NUMBER_SUGGESTIONS = 1
RECOMMENDER_SOLR_PATH = ""
RECOMMENDER_THRESHOLD_FREQUENCY = 1
SQLALCHEMY_BINDS = {u'': u''}
SQLALCHEMY_DATABASE_URI = ''
SQLALCHEMY_ECHO = False
SQLALCHEMY_POOL_SIZE = 1
SQLALCHEMY_MAX_OVERFLOW = 1 # allow to exceptionally grow the pool by N
SQLALCHEMY_POOL_TIMEOUT = 1 # Specifies the connection timeout in seconds for the pool
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_RECORD_QUERIES = False


# added by eb-deploy (over-write config values) from environment
try:
    import os, json, ast
    G = globals()
    for k, v in os.environ.items():
        if k == k.upper() and k in G:
            logging.info("Overwriting constant '%s' old value '%s' with new value '%s'", k, G[k], v)
            try:
                w = json.loads(v)
                G[k] = w
            except:
                try:
                    # Interpret numbers, booleans, etc...
                    G[k] = ast.literal_eval(v)
                except:
                    # String
                    G[k] = v
except:
    pass
