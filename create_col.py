## milvus conncetion
from pymilvus import (utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,list_collections)

print("start connecting to Milvus")
connections.connect("default", host="localhost", port="19530")

## create milvus collection
fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True, auto_id=False),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR,max_length=200),
            FieldSchema(name="bm25_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=8192),
         ]
schema = CollectionSchema(fields, "")
col_name = "bm25_test"
bm25_test = Collection(col_name, schema, consistency_level="Strong")
print("------完成bm25_test的创建--------------")
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
bm25_test.create_index("bm25_vector", sparse_index)
print("------完成bm25_vector的索引创建--------------")