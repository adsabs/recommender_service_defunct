from flask import current_app, Blueprint, request
from flask.ext.restful import Resource
from flask.ext.discoverer import advertise
import time
import inspect
import sys

from utils.recommender import get_recommendations

blueprint = Blueprint(
      'recommender',
      __name__,
      static_folder=None,
)

class Recommender(Resource):
    """Return recommender results for a given bibcode"""
    scopes = []
    rate_limit = [1000,60*60*24]
    decorators = [advertise('scopes','rate_limit')]
    def get(self, bibcode):
        try:
            results = get_recommendations(bibcode)
        except Exception, err:
            return {'msg': 'Unable to get results! (%s)' % err}, 500

        if results:
            return results
        else:
            return {'msg': 'No recommendations found'}, 404
