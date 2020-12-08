from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

xformer = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
xformer.save('/tmp/sentence_transformer')

ce = CrossEncoder('sentence-transformers/ce-roberta-base-stsb', max_length=128)

# q_model = torch.quantization.quantize_dynamic(ce.model, dtype=torch.qint8) # quantize for better performance.
# ce.model = q_model

ce.save_pretrained('/tmp/cross_encoder')

