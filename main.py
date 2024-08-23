from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
from bm25milvus import BM25Milvus
import json
import time


if __name__ == '__main__':
    with open('sanguo.json', 'r') as f:
        data = json.load(f)
    corpus = []
    for doc in data[0:4]:
        corpus.append(doc['_source']['snippet'])

    # there are some built-in analyzers for several languages, now we use 'en' for English.
    analyzer = build_default_analyzer(language="zh")

    # Use the analyzer to instantiate the BM25EmbeddingFunction
    bm25_ef = BM25Milvus(analyzer, chunk_size=500)
    # bm25_ef.fit(corpus)
    # bm25_ef.save('0-4.json')

    new_doc = corpus[-1]
    bm25_ef.load('0-3.json')
    start_time = time.time()
    bm25_ef.add_single_doc(new_doc)
    end_time = time.time()
    print('time cost:', end_time-start_time)
    bm25_ef.save('model_addone.json')
