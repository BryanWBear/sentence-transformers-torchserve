
"""
Handler for semantic_search using SentenceTransformer 
"""

from sentence_transformers import SentenceTransformer
import scipy.spatial
import json
import zipfile
from json import JSONEncoder
import numpy as np

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
        print(model_dir)
        try:
            with zipfile.ZipFile(model_dir + '/pytorch_model.bin', 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            with zipfile.ZipFile(model_dir + '/pool.zip', 'r') as zip_ref:
                zip_ref.extractall(model_dir)
        except:
            print('tried unzipping again')
        
        self.embedder = SentenceTransformer(model_dir)
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
        query_embeddings = self.embedder.encode(data)
        return query_embeddings

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

# Tester
'''class Ctx(object):
    pass
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
data = {'corpus':corpus,'queries':queries}
properties = {}
properties["model_dir"] = '/Users/dhaniram_kshirsagar/projects/neo-sagemaker/mms/code/serve/examples/semantic_search'
ctx = Ctx( )
ctx.system_properties = properties
output = handle([{'data':json.dumps(data)}],ctx)
print(output)'''