import sys
import os
PROJECT_HOME = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_HOME)
from flask.ext.testing import TestCase
from flask import request
from flask import url_for, Flask
from models import db, Clusters, Clustering, CoReads, Reads
import unittest
import requests
import time
import app
import json
import httpretty
import numpy as np
import mock


def get_testclusters(n):
    clusters = []
    c1 = Clusters(
        cluster=1,
        members=['a', 'b', 'c', 'd'],
        centroid=[0.1, 0.1, 0.1, 0.1, 0.1]
    )
    c2 = Clusters(
        cluster=2,
        members=['e', 'f', 'g'],
        centroid=[1.1, 1.1, 1.1, 1.1, 1.1]
    )
    if n == 1:
        return c1
    else:
        return [c1, c2]


def get_paperinfo():
    data = []
    for i in range(1, 5):
        cl = Clustering(bibcode='paper_%s' %
                        str(i), vector_low=[float(i) / 10] * 5)
        data.append(cl)
    ar1 = Reads(
        cookie='u1',
        reads=['ppr1'] * 13 + ['ppr2'] * 7
    )
    data.append(ar1)
    ar2 = Reads(
        cookie='u2',
        reads=['ppr2'] * 53 + ['ppr3'] * 17
    )
    data.append(ar2)
    return data


def get_coreads():
    cr = CoReads(
        bibcode='paper_3',
        coreads={'before': [['ppr1', 13], ['ppr2', 7]],
                 'after': [['ppr2', 53], ['ppr3', 17]]},
        readers=['u1', 'u2'],
    )
    return cr


class TestConfig(TestCase):

    '''Check if config has necessary entries'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    def test_config_values(self):
        '''Check if all required config variables are there'''
        required = ["RECOMMENDER_MAX_HITS", "RECOMMENDER_MAX_INPUT",
                    "RECOMMENDER_CHUNK_SIZE", "RECOMMENDER_MAX_NEIGHBORS",
                    "RECOMMENDER_NUMBER_SUGGESTIONS",
                    "RECOMMENDER_THRESHOLD_FREQUENCY",
                    "RECOMMENDER_SOLR_PATH",
                    "RECOMMENDER_CLUSTER_PROJECTION_PATH",
                    "SQLALCHEMY_BINDS", "DISCOVERER_PUBLISH_ENDPOINT",
                    "DISCOVERER_SELF_PUBLISH", "SQLALCHEMY_BINDS"]

        missing = [x for x in required if x not in self.app.config.keys()]
        self.assertTrue(len(missing) == 0)
        # Check if API has an actual value if we have a 'local_config.py'
        # (not available when testing with Travis)
        if os.path.exists("%s/local_config.py" % PROJECT_HOME):
            self.assertTrue(
                self.app.config.get('RECOMMENDER_API_TOKEN', None) != None)


class TestHelperFunctions(TestCase):

    '''Check if the helper functions return expected results'''

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

    def test_mock_query(self):
        '''Check that session mock behaves the way we set it up'''
        expected_attribs = [
            '_sa_instance_state', 'centroid', 'cluster', 'members']
        # Quering the mock for one record should return one Clusters instance
        resp = db.session.query(Clusters).filter(Clusters.cluster == 1).one()
        self.assertEqual(
            sorted(resp.__dict__.keys()), sorted(expected_attribs))
        # Querying the mock for all records should return a lists of Clusters
        res = db.session.query(Clusters).all()
        self.assertEqual(
            [x.__class__.__name__ for x in res], ['Clusters', 'Clusters'])
        self.assertTrue(isinstance(res, list))

    def test_flattener(self):
        '''Test the method that turns lists of lists into a single list'''
        from recommender import flatten
        data = [[[1, 2, 3], (42, None)], [4, 5], [6], 7, (8, 9, 10)]
        expected = [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(flatten(data), expected)

    def test_tuplemerger(self):
        '''Test the method that merges tuples'''
        from recommender import merge_tuples
        l1 = [['a', 1], ['b', 2], ['c', 5]]
        l2 = [['b', 1], ['c', 6], ['d', 1]]
        expected = [('a', 1), ('c', 11), ('b', 3), ('d', 1)]
        self.assertEqual(merge_tuples(l1, l2), expected)

    def test_frequencies(self):
        '''Test the method that generates a frequency distribution'''
        from recommender import get_frequencies
        l = ['a', 'a', 'a', 'b', 'x', 'x']
        expected = [('a', 3), ('x', 2), ('b', 1)]
        self.assertEqual(get_frequencies(l), expected)


class TestDataRetrieval(TestCase):

    '''Check if methods return expected results'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    @httpretty.activate
    def test_normalized_keywords(self):
        '''Test to see if normalized keywords method behaves as expected'''
        from recommender import get_normalized_keywords
        # When normalized keywords from the AST collection are supplied, they
        # should be returned
        expected_keywords = ["aberration", "ablation", "absorption"]
        mockdata = [
            {'id': '1', 'bibcode': 'a',
             'keyword_norm': expected_keywords + ["x", "y", "z"]}]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
                       "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))

        results = get_normalized_keywords('bibcode')
        self.assertTrue('Results' in results)
        self.assertEqual(results['Results'], expected_keywords)

        # When other keywords (i.e. not in AST collection) are supplied,
        # nothing gets returned (Error)
        mockdata = [
            {'id': '1', 'bibcode': 'a', 'keyword_norm': ["x", "y", "z"]}]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
                       "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))

        results = get_normalized_keywords('bibcode')
        expected = {'Status Code': '200', 'Error': 'Unable to get results!',
                    'Error Info': 'No or unusable keywords in data'}
        self.assertEqual(results, expected)

        # When Solr is unavailable, an error is returned
        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=500,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
                       "wt":"json", "q":"*"}},
            "response":{"numFound":0,"start":0,"docs":[]
            }}""")

        results = get_normalized_keywords('bibcode')
        self.assertTrue('Error' in results)
        self.assertTrue('connection error' in results['Error'])


class TestArticleDataRetrieval(TestCase):

    '''Check if methods return expected results'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    @httpretty.activate
    def test_article_data(self):
        '''Test to see if article data gets returned properly'''
        from recommender import get_article_data
        # By default this method should return a list of dictionaries that
        # includes references
        expected_keywords = ["aberration", "ablation", "absorption"]
        mockdata = [
            {'id': '1', 'bibcode': 'a', 'title': [
                'a_title'], 'first_author':'a_author', 'reference':['x', 'z']},
            {'id': '2', 'bibcode': 'b', 'title': [
                'b_title'], 'first_author':'b_author', 'reference':['d', 'x']},
        ]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
                       "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))

        results = get_article_data('bibcode')
        expected = [{u'first_author': u'a_author',
                     u'bibcode': u'a', u'id': u'1',
                     u'reference': [u'x', u'z'],
                     u'title': [u'a_title']}, {u'first_author': u'b_author',
                                               u'bibcode': u'b', u'id': u'2',
                                               u'reference': [u'd', u'x'],
                                               u'title': [u'b_title']}]
        self.assertEqual(results, expected)

        # When 'check_references=False' is supplied, we should get a dictionary
        # with authors and titles
        results = get_article_data('bibcode', check_references=False)
        expected = {u'a': {'author': u'a_author,+', 'title': u'a_title'},
                    u'b': {'author': u'b_author,+', 'title': u'b_title'}}
        self.assertEqual(results, expected)

        # When Solr is unavailable, an error is returned
        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=500,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":0,"start":0,"docs":[]
            }}""")

        results = get_article_data('bibcode')
        self.assertTrue('Error' in results)
        self.assertTrue('connection error' in results['Error'])


class TestCitationDataRetrieval(TestCase):

    '''Check if methods return expected results'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    @httpretty.activate
    def test_citation_data(self):
        '''Test to see if citation data gets returned properly'''
        from recommender import get_citing_papers
        # This method should return a dictionary where the
        # key 'Results' holds all the citations
        # for the input bibcodes
        expected_keywords = ["aberration", "ablation", "absorption"]
        mockdata = [
            {'id': '1', 'bibcode': 'a', 'title': [
                'a_title'], 'first_author':'a_author', 'citation':['x', 'z']},
            {'id': '2', 'bibcode': 'b', 'title': [
                'b_title'], 'first_author':'b_author', 'citation':['d', 'x']},
        ]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))

        results = get_citing_papers(bibcodes=['a', 'b'])
        self.assertTrue('Results' in results)
        self.assertEqual(results['Results'], [u'x', u'z', u'd', u'x'])

        # When Solr is unavailable, an error is returned
        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=500,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":0,"start":0,"docs":[]
            }}""")

        results = get_citing_papers(bibcodes=['a', 'b'])
        self.assertTrue('Error' in results)
        self.assertTrue('connection error' in results['Error'])


class TestVectorCreation(TestCase):

    '''Check if methods return expected results'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    @httpretty.activate
    def test_create_paper_vector(self):
        '''Test to see if paper vector get properly created'''
        from recommender import get_normalized_keywords
        from recommender import make_paper_vector
        # For a paper with AST normalized keywords, we should get a non-trivial
        # vector
        expected_keywords = ["aberration", "ablation", "absorption"]
        mockdata = [
            {'id': '1', 'bibcode': 'a',
             'keyword_norm': expected_keywords + ["x", "y", "z"]}]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))

        vector = make_paper_vector('bibcode')
        # The vector should be a list of floats
        self.assertTrue(all(isinstance(x, float) for x in vector))
        # The vector elements should sum up to 1.0
        self.assertTrue(sum(vector) == 1.0)
        # By design we have only the first three
        # normalized keywords: only the first
        # three positions of the vector should be non-zero and equal to 1/3
        expected = [1 / float(3), 1 / float(3), 1 / float(3)]
        self.assertEqual(vector[:3], expected)

        # If a paper has no keywords or no normalized keywords, an error is
        # returned
        mockdata = [
            {'id': '1', 'bibcode': 'a', 'keyword_norm': ["x", "y", "z"]}]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))

        vector = make_paper_vector('bibcode')
        self.assertTrue('Error' in vector)
        self.assertEqual(vector, {'Status Code': '200',
                                  'Error Info':
                                  'No or unusable keywords in data',
                                  'Error': 'Unable to get results!'})


class TestProjection(TestCase):

    '''Check if methods return expected results'''

    def create_app(self):
        '''Create the wsgi application'''
        app_ = app.create_app()
        return app_

    def test_project_paper(self):
        '''Test to see if paper vector gets properly
           projected into a given cluster'''
        from recommender import project_paper
        # First we test projecting a paper in the 100-dimensional
        # space, which corresponds
        # with cluster "-1". This projection happens in the beginning,
        # just after we established the initial paper vector based on
        # its normalized keywords
        vector = [0.0] * 993
        vector[0] = 1 / float(3)
        vector[1] = 1 / float(3)
        vector[2] = 1 / float(3)
        cluster_vector = project_paper(vector)
        # The vector should be a list of floats
        self.assertTrue(all(isinstance(x, float) for x in cluster_vector))
        self.assertTrue(type(cluster_vector).__module__ == np.__name__)
        np.testing.assert_almost_equal(
            sum(list(cluster_vector)), 0.00166657306484, 10)

        # Now we test a project of a paper within a specific cluster, i.e. from
        # 100 to 5 dimensions
        vector = [0.0] * 100
        vector[0] = 1 / float(3)
        vector[1] = 1 / float(3)
        vector[2] = 1 / float(3)
        cluster_vector = project_paper(vector, pcluster=1)
        # The vector should be a list of floats
        self.assertTrue(all(isinstance(x, float) for x in cluster_vector))
        self.assertTrue(type(cluster_vector).__module__ == np.__name__)
        expected = np.array(
            [0.37929732, -0.07042716,  0.10629903,  0.11877967, -0.13284801])
        np.testing.assert_array_almost_equal(cluster_vector, expected, 8)

    def test_find_paper_cluster(self):
        '''Test to see if we find the expected cluster'''
        from recommender import find_paper_cluster
        pvec = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # If we have a paper that is already in a cluster,
        # the 'one' method should find it
        # Paper 'a' is in cluster 1
        bibcode = 'a'
        cluster = find_paper_cluster(pvec, bibcode)
        self.assertTrue(cluster == 1)
        # Paper 'z' is in no cluster, but it's vector is closest to the
        # centroid of cluster 2
        bibcode = 'z'
        cluster = find_paper_cluster(pvec, bibcode)

        self.assertTrue(cluster == 2)

    def test_find_closest_cluster_papers(self):
        '''Test to see if we find the closest papers in a cluster
           for a given paper and cluster'''
        from recommender import find_closest_cluster_papers
        # We only want the one closest paper for testing
        self.app.config['RECOMMENDER_MAX_NEIGHBORS'] = 1
        cluster = 1
        pvec = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        result = find_closest_cluster_papers(cluster, pvec)
        # Since our test paper has the exact same vector as
        # 'paper_3' in the mock data the returned list should
        # have length 1 with 'paper_3' as its entry
        self.assertTrue(len(result) == 1)
        self.assertTrue(result[0] == 'paper_3')

    @httpretty.activate
    def test_find_recommendations(self):
        '''Test to see if recommendations are returned properly'''
        from recommender import find_recommendations
        # To test this method we need both the mock for PostgreSQL
        # and the override for the Solr query (for 'get_article_data').
        # The Solr query needs to return the references and citation counts
        # for the papers found in the co-reads
        mockdata = [
            {'id': '1', 'bibcode': 'ppr1', 'reference': [
                'r1', 'r2'], 'citation':['c1'], 'citation_count':1},
            {'id': '2', 'bibcode': 'ppr2', 'reference': [
                'r2', 'r3'], 'citation':['c1', 'c2', 'c3'],
             'citation_count':3},
            {'id': '2', 'bibcode': 'ppr3', 'reference': [
                'r2', 'r3'], 'citation':['c2', 'c3'], 'citation_count':2}
        ]

        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))
        # Now that we've set up all mock data, time to test!
        input_papers = ['paper_3']
        result = find_recommendations(input_papers)
        expected_recommendations = [
            'Field definitions:', 'paper_3', 'ppr1', 'ppr2', 'ppr2',
            u'ppr1', u'c1', u'r2', u'ppr2']
        self.assertEqual(sorted(result), sorted(expected_recommendations))

    @httpretty.activate
    def test_everything(self):
        '''Test to see if calling everything sequentially works'''
        from recommender import get_recommendations
        # To test this method we need both the mock for PostgreSQL
        # and the override for the Solr query (for 'get_article_data').
        # The Solr query needs to return the references and citation counts
        # for the papers found in the co-reads
        # The mock data below takes everything into account that was tested
        # before
        mockdata = [
            {'id': '1', 'bibcode': 'ppr1',
             'first_author': 'au_ppr1', 'title': [
                 'ttl_ppr1'], 'reference':['r1', 'r2'], 'citation':['c1'],
             'citation_count':1},
            {'id': '2', 'bibcode': 'ppr2', 'first_author': 'au_ppr2',
             'title': ['ttl_ppr2'], 'reference':['r2', 'r3'],
             'citation':['c1', 'c2', 'c3'], 'citation_count':3},
            {'id': '3', 'bibcode': 'ppr3', 'first_author': 'au_ppr3',
             'title': ['ttl_ppr3'], 'reference':['r2', 'r3'],
             'citation':['c2', 'c3'],
             'citation_count':2},
            {'id': '4', 'bibcode': 'foo', 'keyword_norm': [
                "aberration", "ablation", "absorption"]},
            {'id': '5', 'bibcode': 'paper_3',
                'first_author': 'au_paper3', 'title': ['ttl_paper_3']},
            {'id': '6', 'bibcode': 'c1',
                'first_author': 'au_c1', 'title': ['ttl_c1']},
            {'id': '7', 'bibcode': 'r2',
                'first_author': 'au_r2', 'title': ['ttl_r2']}
        ]
        httpretty.register_uri(
            httpretty.GET, self.app.config.get('RECOMMENDER_SOLR_PATH'),
            content_type='application/json',
            status=200,
            body="""{
            "responseHeader":{
            "status":0, "QTime":0,
            "params":{ "fl":"reference,citation", "indent":"true",
            "wt":"json", "q":"*"}},
            "response":{"numFound":10456930,"start":0,"docs":%s
            }}""" % json.dumps(mockdata))
        # With the mock data the following recommendations should get generated
        expected_recommendations = {'paper': 'a',
                                    'recommendations': [
                                       {'bibcode': 'ppr2',
                                        'author': u'au_ppr2,+',
                                        'title': u'ttl_ppr2'},
                                       {'bibcode': 'ppr2',
                                        'author': u'au_ppr2,+',
                                        'title': u'ttl_ppr2'},
                                       {'bibcode': u'ppr1',
                                        'author': u'au_ppr1,+',
                                        'title': u'ttl_ppr1'},
                                       {'bibcode': u'c1',
                                        'author': u'au_c1,+',
                                        'title': u'ttl_c1'},
                                       {'bibcode': u'r2',
                                        'author': u'au_r2,+',
                                        'title': u'ttl_r2'},
                                       {'bibcode': u'ppr2',
                                        'author': u'au_ppr2,+',
                                        'title': u'ttl_ppr2'}]}
        # Generate the recommendations
        recommendations = get_recommendations('a')
        # Do the final check
        self.assertEqual(recommendations, expected_recommendations)

if __name__ == '__main__':
    unittest.main()
