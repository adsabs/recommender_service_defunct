from flask import current_app, request
from flask.ext.restful import Resource
from flask.ext.discoverer import advertise
from recommender import get_recommendations


class Recommender(Resource):

    """Return recommender results for a given bibcode"""
    scopes = []
    rate_limit = [1000, 60 * 60 * 24]
    decorators = [advertise('scopes', 'rate_limit')]

    def get(self, bibcode):
        try:
            results = get_recommendations(bibcode)
        except Exception, err:
            return {'msg': 'Unable to get results! (%s)' % err}, 500

        if 'Error' in results:
            return results, results['Status Code']
        else:
            return results
