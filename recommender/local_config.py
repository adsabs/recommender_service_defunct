SQLALCHEMY_METRICS_DATABASE_URI = 'postgresql+psycopg2://metrics:metrics@localhost:5432/metrics'
SQLALCHEMY_RECOMMENDER_DATABASE_URI = 'postgresql+psycopg2://recommender:recommender@localhost:5432/recommender'
SOLRQUERY_URL = 'http://localhost:9000/solr/collection1/select'
SQLALCHEMY_BINDS = {
    'metrics': SQLALCHEMY_METRICS_DATABASE_URI,
    'recommender': SQLALCHEMY_RECOMMENDER_DATABASE_URI
}
