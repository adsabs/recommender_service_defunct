import simplejson as json
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.dialects import postgresql
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class AlchemyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data) # this will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)

class Reads(db.Model):
    __tablename__='reads'
    __bind_key__ ='recommender'
    id = Column(Integer,primary_key=True)
    cookie = Column(String,nullable=False,index=True)
    reads = Column(postgresql.ARRAY(String))

class CoReads(db.Model):
    __tablename__='coreads'
    __bind_key__ ='recommender'
    id = Column(Integer,primary_key=True)
    bibcode = Column(String,nullable=False,index=True)
    coreads = Column(postgresql.JSON)
    readers = Column(postgresql.ARRAY(String))

class Clustering(db.Model):
    __tablename__='clustering'
    __bind_key__ ='recommender'
    id = Column(Integer,primary_key=True)
    bibcode = Column(String,nullable=False,index=True)
    cluster = Column(Integer)
    vector  = Column(postgresql.ARRAY(Float))
    vector_low = Column(postgresql.ARRAY(Float))

class Clusters(db.Model):
    __tablename__='clusters'
    __bind_key__ ='recommender'
    id = Column(Integer,primary_key=True)
    cluster = Column(Integer,index=True)
    members  = Column(postgresql.ARRAY(String))
    centroid = Column(postgresql.ARRAY(Float))
