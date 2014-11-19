'''
Created on Nov 3, 2014

@author: ehenneken
'''
import os
import re
import sys
import time
from datetime import datetime
import random as rndm
import simplejson as json
from itertools import groupby
import urllib
import numpy as np 
import operator
import cPickle
from .definitions import ASTkeywords
from multiprocessing import Process, Queue, cpu_count
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.dialects import postgresql
from flask import current_app
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

_basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

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

def get_before_after(item,list):
    '''
    For a given list, given an item, give the items
    directly before and after that given item
    '''
    idx = list.index(item)
    try:
        before = list[idx-1]
    except:
        before = "NA"
    try:
        after = list[idx+1]
    except:
        after = "NA"
    return [before,after]

def get_frequencies(l):
    '''
    For a list of items, return a list of tuples, consisting of
    unique items, augemented with their frequency in the original list
    '''
    tmp = [(k,len(list(g))) for k, g in groupby(sorted(l))]
    return sorted(tmp, key=operator.itemgetter(1),reverse=True)[:100]

def make_date(datestring):
    '''
    Turn an ADS publication data into an actual date
    '''
    pubdate = map(lambda a: int(a), datestring.split('-'))
    if pubdate[1] == 0:
        pubdate[1] = 1
    return datetime(pubdate[0],pubdate[1],1)

class AlchemyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data) # this will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)

# Data Model Definitions

class MetricsModel(db.Model):
  __tablename__='metrics'
  __bind_key__ = 'metrics'
  id = Column(Integer,primary_key=True)
  bibcode = Column(String,nullable=False,index=True)
  refereed = Column(Boolean)
  rn_citations = Column(postgresql.REAL)
  rn_citation_data = Column(postgresql.JSON)
  rn_citations_hist = Column(postgresql.JSON)
  downloads = Column(postgresql.ARRAY(Integer))
  reads = Column(postgresql.ARRAY(Integer))
  an_citations = Column(postgresql.REAL)
  refereed_citation_num = Column(Integer)
  citation_num = Column(Integer)
  citations = Column(postgresql.ARRAY(String))
  refereed_citations = Column(postgresql.ARRAY(String))
  author_num = Column(Integer)
  an_refereed_citations = Column(postgresql.REAL)
  modtime = Column(DateTime)

class Reads(db.Model):
    __tablename__='reads'
    __bind_key__ = 'recommender'
    id = Column(Integer,primary_key=True)
    cookie = Column(String,nullable=False,index=True)
    reads = Column(postgresql.ARRAY(String))

class Clustering(db.Model):
    __tablename__='clustering'
    __bind_key__ = 'recommender'
    id = Column(Integer,primary_key=True)
    bibcode = Column(String,nullable=False,index=True)
    cluster = Column(Integer)
    vector  = Column(postgresql.ARRAY(Float))
    vector_low = Column(postgresql.ARRAY(Float))

class Clusters(db.Model):
    __tablename__='clusters'
    __bind_key__ = 'recommender'
    id = Column(Integer,primary_key=True)
    cluster = Column(Integer,index=True)
    members  = Column(postgresql.ARRAY(String))
    centroid = Column(postgresql.ARRAY(Float))

# Data retrieval
class DistanceHarvester(Process):
    '''
    Class to find the distance between a given document, represented
    by its document vector, and a cluster document
    '''
    def __init__(self, task_queue, result_queue):
        Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.session = db.session()
    def run(self):
        while True:
            data = self.task_queue.get()
            if data is None:
                break
            try:
                result = self.session.query(Clustering).filter(Clustering.bibcode==data[0]).one()
                clustering_data = json.dumps(result, cls=AlchemyEncoder)
                cvect = np.array(clustering_data['vector_low'])
                dist = np.linalg.norm(data[1]-cvect)
            except:
                dist = 999999
            self.result_queue.put((data[0],dist))
        return

class CitationListHarvester(Process):
    """
    Class to allow parallel retrieval of citation data from Mongo
    """
    def __init__(self, task_queue, result_queue):
        Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.session = db.session()
    def run(self):
        while True:
            bibcode = self.task_queue.get()
            if bibcode is None:
                break
            try:
                result = self.session.query(MetricsModel).filter(MetricsModel.bibcode==bibcode).one()
                try:
                    citations = result.citations
                except:
                    citations = []
                self.result_queue.put({'citations':citations})
            except:
                self.result_queue.put({'citations':[]})
        return

def get_normalized_keywords(bibc):
    '''
    For a given publication, construct a list of normalized keywords of this
    publication and its references
    '''
    keywords = []
    q = 'bibcode:%s or references(bibcode:%s)' % (bibc,bibc)
    try:
        # Get the information from Solr
        params = {'wt':'json', 'q':q, 'fl':'keyword_norm', 'rows': current_app.config['MAX_HITS']}
        query_url = current_app.config['SOLRQUERY_URL'] + "/?" + urllib.urlencode(params)
        resp = current_app.client.session.get(query_url).json()
    except SolrQueryError, e:
        app.logger.error("Solr keywords query for %s blew up (%s)" % (bibc,e))
        raise
    for doc in resp['response']['docs']:
        try:
            keywords += map(lambda a: a.lower(), doc['keyword_norm'])
        except:
            pass
    return filter(lambda a: a in ASTkeywords, keywords)

def get_article_data(biblist, check_references=True):
    '''
    Get basic article metadata for a list of bibcodes
    '''
    list = " OR ".join(map(lambda a: "bibcode:%s"%a, biblist))
    q = '%s' % list
    fl= 'bibcode,title,first_author,keyword_norm,reference,citation_count,pubdate'
    try:
        # Get the information from Solr
        params = {'wt':'json', 'q':q, 'fl':fl, 'sort':'pubdate desc, bibcode desc', 'rows': current_app.config['MAX_HITS']}
        query_url = current_app.config['SOLRQUERY_URL'] + "/?" + urllib.urlencode(params)
        resp = current_app.client.session.get(query_url).json()
    except SolrQueryError, e:
        app.logger.error("Solr article data query for %s blew up (%s)" % (str(biblist),e))
        raise
    results = resp['response']['docs']
    if check_references:
        results = filter(lambda a: 'reference' in a, results)
        return results
    else:
        data_dict = {}
        for doc in results:
            title = 'NA'
            if 'title' in doc: title = doc['title'][0]
            author = 'NA'
            if 'first_author' in doc: author = "%s,+"%doc['first_author'].split(',')[0]
            data_dict[doc['bibcode']] = {'title':title, 'author':author}
        return data_dict

def get_citing_papers(**args):
    # create the queues
    tasks = Queue()
    results = Queue()
    # how many threads are there to be used
    if 'threads' in args:
        threads = args['threads']
    else:
        threads = cpu_count()
    bibcodes = args.get('bibcodes',[])
    # initialize the "harvesters" (each harvester get the citations for a bibcode)
    harvesters = [ CitationListHarvester(tasks, results) for i in range(threads)]
    # start the harvesters
    for b in harvesters:
        b.start()
    # put the bibcodes in the tasks queue
    num_jobs = 0
    for bib in bibcodes:
        tasks.put(bib)
        num_jobs += 1
    # add some 'None' values at the end of the tasks list, to faciliate proper closure
    for i in range(threads):
        tasks.put(None)
    # gather all results into one citation dictionary
    cit_list = []
    while num_jobs:
        data = results.get()
        if 'Exception' in data:
            raise PostgresQueryError, data
        cit_list += data.get('citations',[])
        num_jobs -= 1
    return cit_list

#   
# Helper Functions: Data Processing
def make_paper_vector(bibc):
    '''
    Given a bibcode, retrieve the list of normalized keywords for this publication AND
    its references. Then contruct a vector of normalized frequencies. This is an ordered
    vector, i.e. the first entry is for the first normalized keyword etc etc etc
    '''
    data = get_normalized_keywords(bibc)
    if len(data) == 0:
        return []
    freq = dict((ASTkeywords.index(x), float(data.count(x))/float(len(data))) for x in data)
    FreqVec = [0.0]*len(ASTkeywords)
    for i in freq.keys():
        FreqVec[i] = freq[i]
    return FreqVec

def project_paper(pvector,pcluster=None):
    '''
    If no cluster is specified, this routine projects a paper vector (with normalized frequencies
    for ALL normalized keywords) onto the reduced 100-dimensional space. When a cluster is specified
    the this is a cluster-specific projection to further reduce the dimensionality to 5 dimensions
    '''
    if not pcluster:
        pcluster = -1
    matrix_file = "%s/%s/clusterprojection_%s.mat.npy" % (_basedir,current_app.config['CLUSTER_PROJECTION_PATH'], pcluster)
    try:
        projection = np.load(matrix_file)
    except Exeption,err:
        sys.stderr.write('Failed to load projection matrix for cluster %s (%s)'%(pclust,err))
    PaperVector = np.array(pvector)
    try:
        coords = np.dot(PaperVector,projection)
    except:
        coords = []
    return coords

def find_paper_cluster(pvec,bibc):
    '''
    Given a paper vector of normalized keyword frequencies, reduced to 100 dimensions, find out
    to which cluster this paper belongs
    '''
    session = db.session()
    try:
        res = session.query(Clusters).filter(Clusters.members.any(bibc)).one()
        cluster_data = json.dumps(result, cls=AlchemyEncoder)
    except:
        res = None
    if res:
        return cluster_data['cluster']

    min_dist = 9999
    res = session.query(Clusters).all()
    clusters = json.loads(json.dumps(res, cls=AlchemyEncoder))
    for entry in clusters:
        centroid = entry['centroid']
        dist = np.linalg.norm(pvec-np.array(centroid))
        if dist < min_dist:
            cluster = entry['cluster']
        min_dist = min(dist, min_dist)
    return str(cluster)

def find_closest_cluster_papers(pcluster,vec):
    '''
    Given a cluster and a paper (represented by its vector), which are the
    papers in the cluster closest to this paper?
    '''
    session = db.session()
    result = session.query(Clusters).filter(Clusters.cluster==int(pcluster)).one()
    res = json.loads(json.dumps(result, cls=AlchemyEncoder))
    # We will now calculate the distances of the new paper all cluster members
    # This is done in parallel using the DistanceHarvester
    threads = cpu_count()
    tasks = Queue()
    results =Queue()
    harvesters = [DistanceHarvester(tasks,results) for i in range(threads)]
    for b in harvesters:
        b.start()
    num_jobs = 0
    for paper in res['members']:
        tasks.put((paper,vec))
        num_jobs += 1
    for i in range(threads):
        tasks.put(None)
    distances = []
    while num_jobs:
        data = results.get()
        distances.append(data)
        num_jobs -= 1
    d = sorted(distances, key=operator.itemgetter(1),reverse=False)

    return map(lambda a: a[0],d[:current_app.config['MAX_NEIGHBORS']])

def find_recommendations(G,remove=None):
    '''Given a set of papers (which is the set of closest papers within a given
    cluster to the paper for which recommendations are required), find recommendations.'''
    # get all reads series by frequent readers who read
    # any of the closest papers (stored in G)
    session = db.session()
    res = []
    for paper in G:
        result = session.query(Reads).filter(Reads.reads.any(paper)).all()
        res += json.loads(json.dumps(result, cls=AlchemyEncoder))
    # lists to record papers read just before and after a paper
    # was read from those closest papers, and those to calculate
    # associated frequency distributions
    before = []
    BeforeFreq = []
    after  = []
    AfterFreq = []
    # list to record papers read by people who read one of the
    # closest papers
    alsoreads = []
    AlsoFreq = []
    # start processing those reads we determined earlier
    for item in res:
        alsoreads += item['reads']
        overlap = list(set(item['reads']) & set(G))
        before_after_reads = map(lambda a: get_before_after(a, item['reads']), overlap)
        for reads_pair in before_after_reads:
            before.append(reads_pair[0])
            after.append(reads_pair[1])
    # remove all "NA"
    before = filter(lambda a: a != "NA", before)
    after  = filter(lambda a: a != "NA", after)
    # remove (if specified) the paper for which we get recommendations
    if remove:
        alsoreads = filter(lambda a: a != remove, alsoreads)
    # calculate frequency distributions
    BeforeFreq = get_frequencies(before)
    AfterFreq  = get_frequencies(after)
    AlsoFreq  = get_frequencies(alsoreads)
    # get publication data for the top 100 most alsoread papers
    top100 = map(lambda a: a[0], AlsoFreq)
    top100_data = get_article_data(top100)
    # For publications with no citations, Solr docs don't have a citation count
    tmpdata = []
    for item in top100_data:
        if 'citation_count' not in item:
            item.update({'citation_count':0})
        tmpdata.append(item)
    top100_data = tmpdata
    mostRecent = top100_data[0]['bibcode']
    top100_data = sorted(top100_data, key=operator.itemgetter('citation_count'),reverse=True)
    # get the most cited paper from the top 100 most alsoread papers
    MostCited = top100_data[0]['bibcode']
    # get the most papers cited BY the top 100 most alsoread papers
    # sorted by citation
    refs100 = flatten(map(lambda a: a['reference'], top100_data))
    RefFreq = get_frequencies(refs100)
    # get the papers that cite the top 100 most alsoread papers
    # sorted by frequency
    cits100 = get_citing_papers(bibcodes=top100)
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
        raise Exception('make_paper_vector: failed to make paper vector (%s): %s' % (bibcode,str(e)))
    try:
        pvec = project_paper(vec)
    except Exception, e:
        raise Exception('project_paper: failed to project paper vector (%s): %s' % (bibcode,str(e)))
    try:
        pclust = find_paper_cluster(pvec,bibcode)
    except Exception, e:
        raise Exception('find_paper_cluster: failed to find cluster (%s): %s' % (bibcode,str(e)))
    try:
        cvec = project_paper(pvec,pcluster=pclust)
    except Exception, e:
        raise Exception('project_paper: failed to project %s within cluster %s: %s'%(bibcode,pclust, str(e)))
    try:
        close = find_closest_cluster_papers(pclust,cvec)
    except Exception, e:
        raise Exception('find_closest_cluster_papers: failed to find closest cluster papers (%s): %s'%(bibcode,str(e)))
    try:
        R = find_recommendations(close,remove=bibcode)
    except Exception, e:
        raise Exception('find_recommendations: failed to find recommendations. paper: %s, closest: %s, error: %s' % (bibcode,str(closest),str(e)))
    # Get meta data for the recommendations
    try:
        meta_dict = get_article_data(R[1:], check_references=False)
    except Exception, e:
        raise Exception('get_article_data: failed to retrieve article data for recommendations (%s): %s'%(bibcode,str(e)))
    # Filter out any bibcodes for which no meta data was found
    recommendations = filter(lambda a: a in meta_dict, R)

    result = {'paper':bibcode,
              'recommendations':[{'bibcode':x,'title':meta_dict[x]['title'], 
              'author':meta_dict[x]['author']} for x in recommendations[1:]]}

    return result
