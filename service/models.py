from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.dialects import postgresql
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Reads(db.Model):
    __tablename__ = 'reads'
    __bind_key__ = 'recommender'
    id = Column(Integer, primary_key=True)
    cookie = Column(String, nullable=False, index=True)
    reads = Column(postgresql.ARRAY(String))


class CoReads(db.Model):
    __tablename__ = 'coreads'
    __bind_key__ = 'recommender'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String, nullable=False, index=True)
    coreads = Column(postgresql.JSON)
    readers = Column(postgresql.ARRAY(String))


class Clustering(db.Model):
    __tablename__ = 'clustering'
    __bind_key__ = 'recommender'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String, nullable=False, index=True)
    cluster = Column(Integer)
    vector = Column(postgresql.ARRAY(Float))
    vector_low = Column(postgresql.ARRAY(Float))


class Clusters(db.Model):
    __tablename__ = 'clusters'
    __bind_key__ = 'recommender'
    id = Column(Integer, primary_key=True)
    cluster = Column(Integer, index=True)
    members = Column(postgresql.ARRAY(String))
    centroid = Column(postgresql.ARRAY(Float))
