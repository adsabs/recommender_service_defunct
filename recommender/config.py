SECRET_KEY = 'this should be changed'
MAX_HITS = 10000    
MAX_INPUT = 500
CHUNK_SIZE = 100
MAX_NEIGHBORS = 40
NUMBER_SUGGESTIONS = 10
SQLALCHEMY_METRICS_DATABASE_URI = ''
SQLALCHEMY_RECOMMENDER_DATABASE_URI = ''
THRESHOLD_FREQUENCY = 1
SOLR_PATH = 'http://0.0.0.0:9000/solr/select'
CLUSTER_PROJECTION_PATH = 'utils/data/clusters'
#This section configures this application to act as a client, for example to query solr via adsws
CLIENT = {
  'TOKEN': 'we will provide an api key token for this application'
}
# Define the autodiscovery endpoint
DISCOVERER_PUBLISH_ENDPOINT = '/resources'
# Advertise its own route within DISCOVERER_PUBLISH_ENDPOINT
DISCOVERER_SELF_PUBLISH = False
