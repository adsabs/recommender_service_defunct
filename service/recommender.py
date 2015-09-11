'''
Created on Nov 3, 2014

@author: ehenneken
'''
import os
import re
import time
from datetime import datetime
import simplejson as json
from itertools import groupby, chain
from collections import defaultdict
import urllib
import numpy as np
import operator
from definitions import ASTkeywords
from flask import current_app, request
from models import db, Reads, CoReads, Clusters, Clustering, AlchemyEncoder
from client import client

_basedir = os.path.abspath(os.path.dirname(__file__))

# Helper functions
# Data conversion
def flatten(items):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8, 9, 10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for item in items:
        if hasattr(item, '__iter__'):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def merge_tuples(list1, list2):
    merged = defaultdict(int)
    merged.update(list1)
    for key, value in list2:
        merged[key] += value
    return merged.items()


def get_frequencies(l):
    '''
    For a list of items, return a list of tuples, consisting of
    unique items, augemented with their frequency in the original list
    '''
    tmp = [(k, len(list(g))) for k, g in groupby(sorted(l))]
    return sorted(tmp, key=operator.itemgetter(1), reverse=True)[:100]

# Data retrieval


def get_normalized_keywords(bibc):
    '''
    For a given publication, construct a list of normalized keywords of this
    publication and its references
    '''
    keywords = []
    headers = {'X-Forwarded-Authorization':
               request.headers.get('Authorization')}
    q = 'bibcode:%s or references(bibcode:%s)' % (bibc, bibc)
    # Get the information from Solr
    solr_args = {'wt': 'json', 'q': q, 'fl': 'keyword_norm',
                 'rows': current_app.config['RECOMMENDER_MAX_HITS']}
    response = client().get(
        current_app.config.get("RECOMMENDER_SOLR_PATH"),
        params=solr_args, headers=headers)
    if response.status_code != 200:
        return {"Error": "There was a connection error",
                "Error Info": response.text, "Status Code": "500"}
    resp = response.json()
    for doc in resp['response']['docs']:
        try:
            keywords += map(lambda a: a.lower(), doc['keyword_norm'])
        except:
            pass
    keywords = filter(lambda a: a in ASTkeywords, keywords)
    if len(keywords) == 0:
        return {"Error": "Unable to get results!",
                "Error Info": "No or unusable keywords in data",
                "Status Code": "200"}
    else:
        return {"Results": keywords}


def get_article_data(biblist, check_references=True):
    '''
    Get basic article metadata for a list of bibcodes
    '''
    list = " OR ".join(map(lambda a: "bibcode:%s" % a, biblist))
    headers = {'X-Forwarded-Authorization':
               request.headers.get('Authorization')}
    q = '%s' % list
    fl = 'bibcode,title,first_author,keyword_norm,\
          reference,citation_count,pubdate'
    # Get the information from Solr
    solr_args = {'wt': 'json', 'q': q, 'fl': fl,
                 'sort': 'pubdate desc, bibcode desc',
                 'rows': current_app.config['RECOMMENDER_MAX_HITS']}
    response = client().get(
        current_app.config.get("RECOMMENDER_SOLR_PATH"),
        params=solr_args, headers=headers)
    if response.status_code != 200:
        return {"Error": "There was a connection error with Solr",
                "Error Info": response.text, "Status Code": "500"}
    resp = response.json()
    results = resp['response']['docs']
    if check_references:
        results = filter(lambda a: 'reference' in a, results)
        return results
    else:
        data_dict = {}
        for doc in results:
            title = 'NA'
            if 'title' in doc:
                title = doc['title'][0]
            author = 'NA'
            if 'first_author' in doc:
                author = "%s,+" % doc['first_author'].split(',')[0]
            data_dict[doc['bibcode']] = {'title': title, 'author': author}
        return data_dict


def get_citing_papers(**args):
    citations = []
    bibcodes = args.get('bibcodes', [])
    list = " OR ".join(map(lambda a: "bibcode:%s" % a, bibcodes))
    headers = {'X-Forwarded-Authorization':
               request.headers.get('Authorization')}
    q = '%s' % list
    fl = 'citation'
    # Get the information from Solr
    solr_args = {'wt': 'json', 'q': q, 'fl': fl,
                 'sort': 'pubdate desc, bibcode desc',
                 'rows': current_app.config['RECOMMENDER_MAX_HITS']}
    response = client().get(
        current_app.config.get("RECOMMENDER_SOLR_PATH"),
        params=solr_args, headers=headers)
    if response.status_code != 200:
        return {"Error": "There was a connection error with Solr",
                "Error Info": response.text, "Status Code": "500"}
    resp = response.json()
    for doc in resp['response']['docs']:
        if 'citation' in doc:
            citations += doc['citation']
    return {'Results': citations}
#
# Helper Functions: Data Processing


def make_paper_vector(bibc):
    '''
    Given a bibcode, retrieve the list of normalized keywords
    for this publication AND its references. Then contruct a
    vector of normalized frequencies. This is an ordered vector,
    i.e. the first entry is for the first normalized keyword etc etc etc
    '''
    result = get_normalized_keywords(bibc)
    if 'Error' in result:
        return result
    data = result['Results']
    freq = dict(
        (ASTkeywords.index(x),
            float(data.count(x)) / float(len(data))) for x in data)
    FreqVec = [0.0] * len(ASTkeywords)
    for i in freq.keys():
        FreqVec[i] = freq[i]
    return FreqVec


def project_paper(pvector, pcluster=None):
    '''
    If no cluster is specified, this routine projects a paper vector
    (with normalized frequencies for ALL normalized keywords) onto
    the reduced 100-dimensional space. When a cluster is specified
    the this is a cluster-specific projection to further reduce the
    dimensionality to 5 dimensions
    '''
    if pcluster == None:
        pcluster = -1
    matrix_file = "%s/%s/clusterprojection_%s.mat.npy" % (
        _basedir, current_app.config['RECOMMENDER_CLUSTER_PROJECTION_PATH'],
        pcluster)
    try:
        projection = np.load(matrix_file)
    except Exception, err:
        current_app.logger.error(
            'Failed to load projection matrix for cluster %s (%s)' %
            (pclust, err))
    PaperVector = np.array(pvector)
    try:
        coords = np.dot(PaperVector, projection)
    except:
        coords = []
    return coords


def find_paper_cluster(pvec, bibc):
    '''
    Given a paper vector of normalized keyword frequencies, reduced to
    100 dimensions, find out to which cluster this paper belongs
    '''
    # Let us first check if the bibcode at hand already happens to be in one
    # of the clusters
    try:
        res = db.session.query(Clusters).filter(
            Clusters.members.any(bibc)).one()
        cluster = res.cluster
        members = res.members
    except:
        cluster = None
    # Let's double check that the given bibcode is indeed among the cluster
    # members
    if cluster and bibc in members:
        return cluster
    # Apparently not, so now we need to find the cluster where the distance
    # to the cluster centroid is the smallest
    min_dist = 9999
    res = db.session.query(Clusters).all()
    for entry in res:
        centroid = entry.centroid
        dist = np.linalg.norm(pvec - np.array(centroid))
        if dist < min_dist:
            cluster = entry.cluster
        min_dist = min(dist, min_dist)
    return cluster


def find_closest_cluster_papers(pcluster, vec):
    '''
    Given a cluster and a paper (represented by its vector), which are the
    papers in the cluster closest to this paper?
    '''
    # Find the cluster info for the given cluster, in particular the cluster
    # members (identified by their bibcodes)
    cluster_info = db.session.query(Clusters).filter(
        Clusters.cluster == int(pcluster)).one()
    # For each cluster member, retrieve their lower dimensional coordinate
    # tuple, so that we can calculate the distance of the current papers
    # (the coordinates are stored in 'vec')\
    results = db.session.query(Clustering).filter(Clustering.bibcode.in_(tuple(cluster_info.members))).all()
    distances = []
    for result in results:
        try:
            paper = result.bibcode
            pvect = np.array(result.vector_low)
            distance = np.linalg.norm(pvect - vec)
            distances.append((paper, distance))
        except:
            continue
    # All distances have been recorded, now sort them by distance (ascending),
    # and return the appropriate amount
    d = sorted(distances, key=operator.itemgetter(1), reverse=False)
    return map(lambda a: a[0],
               d[:current_app.config['RECOMMENDER_MAX_NEIGHBORS']])


def find_recommendations(G, remove=None):
    '''Given a set of papers (which is the set of closest papers within
       a given cluster to the paper for which recommendations are required),
       find recommendations.'''
    # Get all coreads by frequent readers who read any of the closest papers
    # (stored in G). The coreads consist of frequencies of papers read just
    # before, or just after the paper in the closest papers.
    # The alsoreads are taken to be all the coreads taken together
    BeforeFreq = []
    AfterFreq = []
    readers = []
    for paper in G:
        result = db.session.query(CoReads).filter(
            CoReads.bibcode == paper).first()
        if not result:
            continue
        BeforeFreq = merge_tuples(BeforeFreq, result.coreads['before'])
        AfterFreq = merge_tuples(AfterFreq, result.coreads['after'])
        readers += result.readers
    # Sort the before and after frequency distributions by frequency
    BeforeFreq = sorted(BeforeFreq, key=lambda x: x[1], reverse=True)
    AfterFreq = sorted(AfterFreq, key=lambda x: x[1], reverse=True)
    # Now the the entire set of also-reads
    results = db.session.query(Reads).filter(Reads.cookie.in_(tuple(readers))).all()
    alsoreads = list(chain(*[r.reads for r in results if hasattr(r, 'reads')]))
    # remove (if specified) the paper for which we get recommendations
    if remove:
        alsoreads = filter(lambda a: a != remove, alsoreads)
    # calculate frequency distribution of alsoreads
    AlsoFreq = get_frequencies(alsoreads)
    # get publication data for the top 100 most alsoread papers
    top100 = map(lambda a: a[0], AlsoFreq)
    top100_data = get_article_data(top100)
    if 'Error' in top100_data:
        return top100_data
    # For publications with no citations, Solr docs don't have a citation count
    tmpdata = []
    for item in top100_data:
        if 'citation_count' not in item:
            item.update({'citation_count': 0})
        tmpdata.append(item)
    top100_data = tmpdata
    mostRecent = top100_data[0]['bibcode']
    # Remove the most recent from the AlsoFreqdata
    AlsoFreq = [a for a in AlsoFreq if a[0] != mostRecent]
    # Get data for the top 100 cited papers
    top100_data = sorted(
        top100_data, key=operator.itemgetter('citation_count'), reverse=True)
    # get the most cited paper from the top 100 most alsoread papers
    MostCited = top100_data[0]['bibcode']
    # get the most papers cited BY the top 100 most alsoread papers
    # sorted by citation
    refs100 = flatten(map(lambda a: a['reference'], top100_data))
    RefFreq = get_frequencies(refs100)
    # get the papers that cite the top 100 most alsoread papers
    # sorted by frequency
    data = get_citing_papers(bibcodes=top100)
    if 'Error' in data:
        return data
    cits100 = data['Results']
    CitFreq = get_frequencies(cits100)
    # now we have everything to build the recommendations
    FieldNames = 'Field definitions:'
    Recommendations = []
    Recommendations.append(FieldNames)
    Recommendations.append(G[0])
    Recommendations.append(BeforeFreq[0][0])
    if AfterFreq[0][0] == BeforeFreq[0][0]:
        try:
            Recommendations.append(AfterFreq[1][0])
        except:
            Recommendations.append(AfterFreq[0][0])
    else:
        Recommendations.append(AfterFreq[0][0])
    try:
        Recommendations.append(rndm.choice(AlsoFreq[:10])[0])
    except:
        Recommendations.append(AlsoFreq[0][0])
    Recommendations.append(mostRecent)
    try:
        Recommendations.append(rndm.choice(CitFreq[:10])[0])
    except:
        Recommendations.append(CitFreq[0][0])
    try:
        Recommendations.append(rndm.choice(RefFreq[:10])[0])
    except:
        Recommendations.append(RefFreq[0][0])
    Recommendations.append(MostCited)

    return Recommendations

# The actual recommending functions


def get_recommendations(bibcode):
    '''
    Recommendations for a single bibcode
    '''
    try:
        vec = make_paper_vector(bibcode)
    except Exception, e:
        raise Exception(
            'make_paper_vector: failed to make paper vector (%s): %s' %
            (bibcode, str(e)))
    if 'Error' in vec:
        return vec
    try:
        pvec = project_paper(vec)
    except Exception, e:
        raise Exception(
            'project_paper: failed to project paper vector (%s): %s' %
            (bibcode, str(e)))
    try:
        pclust = find_paper_cluster(pvec, bibcode)
    except Exception, e:
        raise Exception(
            'find_paper_cluster: failed to find cluster (%s): %s' %
            (bibcode, str(e)))
    try:
        cvec = project_paper(pvec, pcluster=pclust)
    except Exception, e:
        raise Exception('project_paper: failed to project %s within \
                        cluster %s: %s' % (bibcode, pclust, str(e)))
    stime=time.time()
    try:
        close = find_closest_cluster_papers(pclust, cvec)
    except Exception, e:
        raise Exception('find_closest_cluster_papers: failed to find \
                        closest cluster papers (%s): %s' % (bibcode, str(e)))
    etime=time.time()
    try:
        R = find_recommendations(close, remove=bibcode)
    except Exception, e:
        raise Exception('find_recommendations: failed to find recommendations. \
                        paper: %s, closest: %s, error: %s' % (
            bibcode, str(close), str(e)))
    # Get meta data for the recommendations
    try:
        meta_dict = get_article_data(R[1:], check_references=False)
    except Exception, e:
        raise Exception(
            'get_article_data: failed to retrieve article data for \
            recommendations (%s): %s' % (bibcode, str(e)))
    # Filter out any bibcodes for which no meta data was found
    recommendations = filter(lambda a: a in meta_dict, R)

    result = {'paper': bibcode,
              'recommendations': [{'bibcode': x,
                                   'title': meta_dict[x]['title'],
                                   'author':meta_dict[x]['author']}
                                  for x in recommendations[1:]]}

    return result
