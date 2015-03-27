import sys, os
PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))
sys.path.append(PROJECT_HOME)
from flask.ext.testing import TestCase
from flask import request
from flask import url_for, Flask
from utils.database import db,Clusters,CoReads
import unittest
import requests
import time
import app
import json
import httpretty
import mock

def get_testclusters(n):
  clusters = []
  c1 = Clusters(
      cluster = 1,
      members = ['a','b','c','d'],
      centroid = [0.1, 0.1, 0.1, 0.1, 0.1]
  )
  c2 = Clusters(
      cluster = 2,
      members = ['e','f','g'],
      centroid = [1.1, 1.1, 1.1, 1.1, 1.1]
  )
  if n==1:
    return c1
  else:
    return [c1,c2]

def get_paperinfo():
  data = []
  for i in range(1,5):
    paper = 'paper_%s' % str(i)
    pvect = [float(i)/10]*5
    data.append(['',paper,'','',pvect])
  return data

def get_coreads():
  cr = CoReads(
    bibcode = 'paper_3',
    coreads = {'before':[['ppr1',13],['ppr2',7]],
               'after':[['ppr2',53],['ppr3',17]]}
    )
  return cr

class TestExpectedResults(TestCase):
  '''Check if the service returns expected results'''
  def create_app(self):
    '''Create the wsgi application'''
    app_ = app.create_app()
    db.session = mock.Mock()
    one = db.session.query.return_value.filter.return_value.one
    fst = db.session.query.return_value.filter.return_value.first
    all = db.session.query.return_value.all
    exe = db.session.execute
    one.return_value = get_testclusters(1)
    fst.return_value = get_coreads()
    all.return_value = get_testclusters(2)
    exe.return_value = get_paperinfo()
    return app_

  @httpretty.activate
  def test_recommendations_200(self):
    '''Test to see if calling the recommender endpoint works for valid data'''
    # To test this method we need both the mock for PostgreSQL
    # and the override for the Solr query (for 'get_article_data').
    # The Solr query needs to return the references and citation counts
    # for the papers found in the co-reads
    # The mock data below takes everything into account that was tested before
    mockdata = [
    {'id':'1','bibcode':'ppr1','first_author':'au_ppr1','title':['ttl_ppr1'],'reference':['r1','r2'],'citation':['c1'],'citation_count':1},
    {'id':'2','bibcode':'ppr2','first_author':'au_ppr2','title':['ttl_ppr2'],'reference':['r2','r3'],'citation':['c1','c2','c3'],'citation_count':3},
    {'id':'3','bibcode':'ppr3','first_author':'au_ppr3','title':['ttl_ppr3'],'reference':['r2','r3'],'citation':['c2','c3'],'citation_count':2},
    {'id':'4','bibcode':'foo','keyword_norm':["aberration","ablation","absorption"]},
    {'id':'5','bibcode':'paper_3','first_author':'au_paper3','title':['ttl_paper_3']},
    {'id':'6','bibcode':'c1','first_author':'au_c1','title':['ttl_c1']},
    {'id':'7','bibcode':'r2','first_author':'au_r2','title':['ttl_r2']}
    ]
    httpretty.register_uri(
            httpretty.GET, self.app.config.get('SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true", "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}"""%json.dumps(mockdata))
    # With the mock data the following recommendations should get generated
    expected_recommendations = [{'bibcode': 'ppr3', 'author': u'au_ppr3,+', 'title': u'ttl_ppr3'}, {'bibcode': 'ppr2', 'author': u'au_ppr2,+', 'title': u'ttl_ppr2'}, {'bibcode': u'ppr1', 'author': u'au_ppr1,+', 'title': u'ttl_ppr1'}, {'bibcode': u'c1', 'author': u'au_c1,+', 'title': u'ttl_c1'}, {'bibcode': u'r2', 'author': u'au_r2,+', 'title': u'ttl_r2'}, {'bibcode': u'ppr2', 'author': u'au_ppr2,+', 'title': u'ttl_ppr2'}]

    url = url_for('recommender.recommender',bibcode='a')
    r = self.client.get(url)
    # The response should have a status code 200
    self.assertTrue(r.status_code == 200)
    # The 'paper' entry should have the value of the input bibcode
    self.assertTrue(r.json.get('paper') == 'a')
    # The 'recommendations' entry should have the expected recommendations
    self.assertEqual(r.json.get('recommendations'), expected_recommendations)

  @httpretty.activate
  def test_recommendations_500(self):
    '''Test to see if calling the recommender endpoint throws a 500 when Solr is not available'''
    # To test this method we need both the mock for PostgreSQL
    # and the override for the Solr query (for 'get_article_data').
    # The Solr query needs to return the references and citation counts
    # for the papers found in the co-reads
    # The mock data below takes everything into account that was tested before
    httpretty.register_uri(
            httpretty.GET, self.app.config.get('SOLR_PATH'),
            content_type='application/json',
            status=500,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true", "wt":"json", "q":"*"}},
            "response":{"numFound":0,"start":0,"docs":[]
            }}""")

    url = url_for('recommender.recommender',bibcode='a')
    r = self.client.get(url)
    # The response should have a status code 500
    self.assertTrue(r.status_code == 500)

  @httpretty.activate
  def test_recommendations_404(self):
    '''Test to see if calling the recommender endpoint works for no keywords'''
    # To test this method we need both the mock for PostgreSQL
    # and the override for the Solr query (for 'get_article_data').
    # The Solr query needs to return the references and citation counts
    # for the papers found in the co-reads
    # The mock data below takes everything into account that was tested before
    mockdata = [
    {'id':'1','bibcode':'ppr1','first_author':'au_ppr1','title':['ttl_ppr1'],'reference':['r1','r2'],'citation':['c1'],'citation_count':1},
    {'id':'2','bibcode':'ppr2','first_author':'au_ppr2','title':['ttl_ppr2'],'reference':['r2','r3'],'citation':['c1','c2','c3'],'citation_count':3},
    {'id':'3','bibcode':'ppr3','first_author':'au_ppr3','title':['ttl_ppr3'],'reference':['r2','r3'],'citation':['c2','c3'],'citation_count':2},
    {'id':'4','bibcode':'foo','keyword_norm':["x","y","z"]},
    {'id':'5','bibcode':'paper_3','first_author':'au_paper3','title':['ttl_paper_3']},
    {'id':'6','bibcode':'c1','first_author':'au_c1','title':['ttl_c1']},
    {'id':'7','bibcode':'r2','first_author':'au_r2','title':['ttl_r2']}
    ]
    httpretty.register_uri(
            httpretty.GET, self.app.config.get('SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true", "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}"""%json.dumps(mockdata))
    # With the mock data the following recommendations should get generated
    expected_recommendations = [{'bibcode': 'ppr3', 'author': u'au_ppr3,+', 'title': u'ttl_ppr3'}, {'bibcode': 'ppr2', 'author': u'au_ppr2,+', 'title': u'ttl_ppr2'}, {'bibcode': u'ppr1', 'author': u'au_ppr1,+', 'title': u'ttl_ppr1'}, {'bibcode': u'c1', 'author': u'au_c1,+', 'title': u'ttl_c1'}, {'bibcode': u'r2', 'author': u'au_r2,+', 'title': u'ttl_r2'}, {'bibcode': u'ppr2', 'author': u'au_ppr2,+', 'title': u'ttl_ppr2'}]

    url = url_for('recommender.recommender',bibcode='a')
    r = self.client.get(url)
    # The response should have a status code 404
    self.assertTrue(r.status_code == 404)
    # The error message should be that no keywords were found
    self.assertTrue(r.json.get('Error') == 'No keywords were found')

if __name__ == '__main__':
  unittest.main()
