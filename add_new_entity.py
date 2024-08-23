import json
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
from bm25milvus import BM25Milvus
from pymilvus import Collection, connections

print("start connecting to Milvus")
connections.connect("default", host="localhost", port="19530")

with open('sanguo.json', 'r') as f:
    data = json.load(f)

analyzer = build_default_analyzer(language="zh")

# Use the analyzer to instantiate the BM25EmbeddingFunction
bm25_ef = BM25Milvus(analyzer, chunk_size=500)
bm25_ef.load('0-3.json')

d = data[3]
_id = d['_id']
file_name = d['_source']['meta_data']['file_name']
snippet = d['_source']['snippet']
new_doc = snippet
bm25_ef.add_single_doc(new_doc)
# bm25_vector = bm25_ef.encode_documents([snippet])
# entity = [[_id],[file_name],bm25_vector, [snippet]]

# bm25_test = Collection("bm25_test", consistency_level="Strong")
# bm25_test.load()
# bm25_test.insert(entity)

vec = bm25_ef.encode_documents([data[2]['_source']['snippet']])
for e in vec:
    print(e)