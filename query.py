from pymilvus import MilvusClient,Collection, connections
import time
from bm25milvus import BM25Milvus
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer

print("start connecting to Milvus")
connections.connect("default", host="localhost", port="19530")

client = MilvusClient(
    uri="http://localhost:19530"
)

collection = Collection("bm25_test")      # Get an existing collection.
collection.load()

analyzer = build_default_analyzer(language="zh")
bm25_ef = BM25Milvus(analyzer, chunk_size=500)
bm25_ef.load('mymodel.json')

queries = ["督邮刘备爆发一场大乱"]
query_embeddings = bm25_ef.encode_queries(queries)

start_time = time.time()
res = client.search(
    collection_name="bm25_test",
    data=query_embeddings,
    anns_field="bm25_vector",
    limit=4,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["id", "file_name", "snippet"]
)
end_time = time.time()
print("total time cost:", end_time-start_time)

for res in res[0]:
    print(res['id'], res['distance'], res['entity']['snippet'])

