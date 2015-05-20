import sys
import os
PROJECT_HOME = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_HOME)
from flask.ext.testing import TestCase
from flask import request
from flask import url_for, Flask
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects import postgresql
from utils.database import CoReads, Clusters, Clustering, AlchemyEncoder
import unittest
import requests
import time
import app
import json
import httpretty
import md5
import numpy as np
import glob


class TestHelperFunctions(TestCase):

    '''Check if the helper functions return expected results'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    def test_keywords(self):
        '''Test whether the list of AST keywords is still sane'''
        from utils.definitions import ASTkeywords
        concat = reduce(lambda x, y: x + y, ASTkeywords, "")
        hash = md5.new()
        hash.update(concat)
        self.assertTrue(len(ASTkeywords) == 993)
        self.assertTrue(hash.hexdigest() == '518410e4816b15594f6e7711ad186e7e')

    def test_clusterprojections(self):
        '''Test whether the cluster projections in data/clusters
           exist and are sane'''
        for i in range(-1, 50):
            P_PATH = self.app.config.get('RECOMMENDER_CLUSTER_PROJECTION_PATH')
            matrix_file = "%s/%s/clusterprojection_%s.mat.npy" % \
                (PROJECT_HOME, P_PATH, i)
            self.assertTrue(os.path.exists(matrix_file) == True)
            projection = np.load(matrix_file)
            if i == -1:
                self.assertEqual(projection.shape, (993, 100))
            else:
                self.assertEqual(projection.shape, (100, 5))


class TestModels(TestCase):

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    def test_data_models(self):
        '''Check that data model for graphics is what we expect'''
        ic = Column(Integer)
        sc = Column(String)
        jc = Column(postgresql.JSON)
        ac = Column(postgresql.ARRAY(Float))
        cc = Column(postgresql.ARRAY(String))
        # test the CoReads model
        cols_expect = map(type, [ic.type, sc.type, jc.type, cc.type])
        self.assertEqual([type(c.type)
                          for c in CoReads.__table__.columns], cols_expect)
        # test the Clustering model
        cols_expect = map(type, [ic.type, sc.type, ic.type, ac.type, ac.type])
        self.assertEqual([type(c.type)
                          for c in Clustering.__table__.columns], cols_expect)
        # test Clusters model
        cols_expect = map(type, [ic.type, ic.type, ac.type, ac.type])
        self.assertEqual([type(c.type)
                          for c in Clusters.__table__.columns], cols_expect)
        # test the class that converts a model to a dictionary
        c = CoReads()
        c.id = 1
        c.bibcode = 'b'
        c.coreads = {}
        c.readers = []
        data = json.loads(json.dumps(c, cls=AlchemyEncoder))
        expected = {u'query': None, u'query_class': None,
                    u'bibcode': u'b', u'id': 1, u'coreads': {}, 'readers': []}
        self.assertEqual(data, expected)

if __name__ == '__main__':
    unittest.main()
