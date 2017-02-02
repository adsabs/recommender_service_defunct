from flask import current_app, request
from flask.ext.restful import Resource
from flask.ext.discoverer import advertise
from recommender import get_recommendations
import time

class Recommender(Resource):

    """Return recommender results for a given bibcode"""
    scopes = []
    rate_limit = [1000, 60 * 60 * 24]
    decorators = [advertise('scopes', 'rate_limit')]

    def get(self, bibcode):
        stime = time.time()
        try:
            results = get_recommendations(bibcode)
        except Exception, err:
            current_app.logger.error('Recommender exception (%s): %s'%(bibcode, err))
            return {'msg': 'Unable to get results! (%s)' % err}, 500

        if 'Error' in results:
            # In case of Solr connection errors, we want to capture more verbose information
            if 'Reponse Code' in results:
                error_code = results['Reponse Code']
                error_msg  = results['Error Info']
                current_app.logger.error('Recommender failed (Solr request failed) for %s: Solr request returned %s (%s)'%(bibcode, error_code, error_msg))
            else:
                current_app.logger.error('Recommender failed (%s): %s'%(bibcode, results.get('Error')))
            return results, results['Status Code']
        else:
            duration = time.time() - stime
            current_app.logger.info('Recommendations for %s in %s user seconds'%(bibcode, duration))
            return results
