
"""
Handler for semantic_search using SentenceTransformer 
"""

from sentence_transformers import CrossEncoder
from transformers import RobertaForSequenceClassification
import scipy.spatial
import json
import zipfile
from json import JSONEncoder
import numpy as np
import torch

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class SematicSearch(object):
    """
    SematicSearch handler class. This handler takes a corpus and query strings
    as input and returns the closest 5 sentences of the corpus for each query sentence based on cosine similarity.
    Ref - https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search.py
    """

    def __init__(self):
        super(SematicSearch, self).__init__()
        self.initialized = False
        self.embedder = None
    
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        self.embedder = CrossEncoder(model_dir)
        self.initialized = True
    
    def preprocess(self, data):
        print(data)
        inputs = data[0].get("data")
        print(inputs)
        if inputs is None:
            inputs = data[0].get("body")
        inputs = inputs.decode('utf-8')
        inputs = json.loads(inputs)
        queries = inputs['queries']

        return queries

    def inference(self, data):
        prediction = self.embedder.predict(data)
        return prediction

    def postprocess(self, data):
        return [json.dumps(data, cls=NumpyArrayEncoder)]


_service = SematicSearch()

def handle(data, context):
    """
    Entry point for SematicSearch handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise Exception("Unable to process input data. " + str(e))