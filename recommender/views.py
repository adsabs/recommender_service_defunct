from flask import current_app, Blueprint, request
from flask.ext.restful import Resource
import time

from utils.recommender import get_recommendations

blueprint = Blueprint(
      'recommender',
      __name__,
      static_folder=None,
)

class Recommender(Resource):
    """"Return recommender results for a given bibcode"""
    scopes = []
    def get(self, bibcode):
       try:
           results = get_recommendations(bibcode)
       except Exception, err:
           return {'msg': 'Unable to get results! (%s)' % err}, 500
       return results

class Resources(Resource):
  '''Overview of available resources'''
  scopes = []
  def get(self):
    func_list = {}
    for rule in current_app.url_map.iter_rules():
      func_list[rule.rule] = {'methods':current_app.view_functions[rule.endpoint].methods,
                              'scopes': current_app.view_functions[rule.endpoint].view_class.scopes,
                              'description': current_app.view_functions[rule.endpoint].view_class.__doc__,
                              }
    return func_list, 200
