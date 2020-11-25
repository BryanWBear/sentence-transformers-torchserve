from sentence_transformers import SentenceTransformer

xformer = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
xformer.save('/tmp/sentence_transformer')