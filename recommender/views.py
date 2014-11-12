from flask import current_app, Blueprint, request
from flask.ext.restful import Resource
import time

from utils.recommender import get_recommendations
# testing
from utils.recommender import make_paper_vector
from utils.recommender import project_paper
from utils.recommender import find_paper_cluster
from utils.recommender import project_paper
from utils.recommender import find_closest_cluster_papers
from utils.recommender import find_recommendations
from utils.recommender import get_recommendations

blueprint = Blueprint(
      'recommender',
      __name__,
      static_folder=None,
)

class Test(Resource):
    """"Only for testing purposes. Will be removedß"""
    scopes = 'oauth:recommender:read'
    def get(self, bibcode, goal):
        if goal == 'vector':
            return make_paper_vector(bibcode)
        elif goal == 'project':
            vec = make_paper_vector(bibcode)
            pvec = project_paper(vec)
            return list(pvec)
        elif goal == 'cluster':
            vec = make_paper_vector(bibcode)
            pvec = project_paper(vec)
            pclust = find_paper_cluster(pvec,bibcode)
            return pclust
        elif goal == 'clustervec':
            vec = make_paper_vector(bibcode)
            pvec = project_paper(vec)
            pclust = find_paper_cluster(pvec,bibcode)
            cvec = project_paper(pvec,pcluster=pclust)
            return list(cvec)
        elif goal == 'closest':
            vec = make_paper_vector(bibcode)
            pvec = project_paper(vec)
            pclust = find_paper_cluster(pvec,bibcode)
            cvec = project_paper(pvec,pcluster=pclust)
            close = find_closest_cluster_papers(pclust,cvec)
            return close
        elif goal == 'recommend':
            vec = make_paper_vector(bibcode)
            pvec = project_paper(vec)
            pclust = find_paper_cluster(pvec,bibcode)
            cvec = project_paper(pvec,pcluster=pclust)
            close = find_closest_cluster_papers(pclust,cvec)
            R = find_recommendations(close,remove=bibcode)
            return R
        elif goal == 'final':
            return get_recommendations(bibcode)
        else:
            return {'msg': 'The goal "%s" is not available'%goal}, 500

class Recommender(Resource):
    """"Return recommender results for a given bibcode"""
    scopes = 'oauth:recommender:read'
    def get(self, bibcode):
       try:
           results = get_recommendations(bibcode)
       except Exception, err:
           return {'msg': 'Unable to get results! (%s)' % err}, 500
       return results

class Resources(Resource):
  '''Overview of available resources'''
  scopes = ['oauth:sample_application:read','oauth_sample_application:logged_in']
  def get(self):
    func_list = {}
    for rule in current_app.url_map.iter_rules():
      func_list[rule.rule] = {'methods':current_app.view_functions[rule.endpoint].methods,
                              'scopes': current_app.view_functions[rule.endpoint].view_class.scopes,
                              'description': current_app.view_functions[rule.endpoint].view_class.__doc__,
                              }
    return func_list, 200

class UnixTime(Resource):
  '''Returns the unix timestamp of the server'''
  scopes = ['oauth:sample_application:read','oauth_sample_application:logged_in']
  def get(self):
    return {'now': time.time()}, 200

class PrintArg(Resource):
  '''Returns the :arg in the route'''
  scopes = ['oauth:sample_application:read','oauth:sample_application:logged_in'] 
  def get(self,arg):
    return {'arg':arg}, 200

class ExampleApiUsage(Resource):
  '''This resource uses the app.client.session.get() method to access an api that requires an oauth2 token, such as our own adsws'''
  scopes = ['oauth:sample_application:read','oauth:sample_application:logged_in','oauth:api:search'] 
  def get(self):
    r = current_app.client.session.get('http://api.adslabs.org/v1/search')
    try:
      r = r.json()
      return {'response':r, 'api-token-which-should-be-kept-secret':current_app.client.token}, 200
    except: #For the moment, 401s are not JSON encoded; this will be changed in the future
      r = r.text
      return {'raw_response':r, 'api-token-which-should-be-kept-secret':current_app.client.token}, 501
