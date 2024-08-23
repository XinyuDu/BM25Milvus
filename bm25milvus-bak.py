from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
from milvus_model.sparse import BM25EmbeddingFunction
from pathlib import Path
import json
from typing import Dict, List, Optional
from milvus_model.sparse.bm25.tokenizers import Analyzer
from collections import defaultdict
import logging
import requests
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class BM25Milvus(BM25EmbeddingFunction):
    def __init__(
        self,
        analyzer: Analyzer = None,
        corpus: Optional[List] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        num_workers: Optional[int] = None,):
        super().__init__(analyzer, corpus, k1, b, epsilon, num_workers)
        self.term_document_frequencies = defaultdict(int)

    def _compute_statistics(self, corpus: List[str]):
        self.term_document_frequencies = defaultdict(int)
        total_word_count = 0
        for document in corpus:
            total_word_count += len(document)

            frequencies = defaultdict(int)
            for word in document:
                frequencies[word] += 1

            for word, _ in frequencies.items():
                self.term_document_frequencies[word] += 1
            self.corpus_size += 1
        self.avgdl = total_word_count / self.corpus_size

    def _rebuild(self, corpus: List[str]):
        self._clear()
        corpus = self._tokenize_corpus(corpus)
        self._compute_statistics(corpus)
        self._calc_idf()
        self._calc_term_indices()


    def save(self, path: str):
        bm25_params = {}
        bm25_params["version"] = "v1"
        bm25_params["corpus_size"] = self.corpus_size
        bm25_params["avgdl"] = self.avgdl
        bm25_params["idf_word"] = [None for _ in range(len(self.idf))]
        bm25_params["idf_value"] = [None for _ in range(len(self.idf))]
        for word, values in self.idf.items():
            bm25_params["idf_word"][values[1]] = word
            bm25_params["idf_value"][values[1]] = values[0]

        bm25_params["k1"] = self.k1
        bm25_params["b"] = self.b
        bm25_params["epsilon"] = self.epsilon

        bm25_params['term_document_frequencies'] = self.term_document_frequencies

        with Path(path).open("w", encoding='utf8') as json_file:
            json.dump(bm25_params, json_file, ensure_ascii=False)

    def load(self, path: Optional[str] = None):
        default_meta_filename = "bm25_msmarco_v1.json"
        default_meta_url = "https://github.com/milvus-io/pymilvus-assets/releases/download/v0.1-bm25v1/bm25_msmarco_v1.json"
        if path is None:
            logger.info(f"path is None, using default {default_meta_filename}.")
            if not Path(default_meta_filename).exists():
                try:
                    logger.info(
                        f"{default_meta_filename} not found, start downloading from {default_meta_url} to ./{default_meta_filename}."
                    )
                    response = requests.get(default_meta_url, timeout=30)
                    response.raise_for_status()
                    with Path(default_meta_filename).open("wb") as f:
                        f.write(response.content)
                    logger.info(f"{default_meta_filename} has been downloaded successfully.")
                except requests.exceptions.RequestException as e:
                    error_message = f"Failed to download the file: {e}"
                    raise RuntimeError(error_message) from e
            path = default_meta_filename
        try:
            with Path(path).open(encoding='utf8') as json_file:
                bm25_params = json.load(json_file)
        except OSError as e:
            error_message = f"Error opening file {path}: {e}"
            raise RuntimeError(error_message) from e
        self.corpus_size = bm25_params["corpus_size"]
        self.avgdl = bm25_params["avgdl"]
        self.idf = {}
        for i in range(len(bm25_params["idf_word"])):
            self.idf[bm25_params["idf_word"][i]] = [bm25_params["idf_value"][i], i]
        self.k1 = bm25_params["k1"]
        self.b = bm25_params["b"]
        self.epsilon = bm25_params["epsilon"]
        self.term_document_frequencies = bm25_params['term_document_frequencies']

    def _calc_idf(self):
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in self.term_document_frequencies.items():
            try:
                idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            except:
                print(word, self.corpus_size, freq)
            if word not in self.idf:
                self.idf[word] = [0.0, 0]
            self.idf[word][0] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word][0] = eps
    def add_single_doc(self, doc):
        terms = self.analyzer(doc)
        doc_len = len(terms)
        terms = list(set(terms))
        self.avgdl = (self.avgdl * self.corpus_size + doc_len)/(self.corpus_size+1)
        self.corpus_size += 1
        for term in terms:
            if term in self.idf:
                self.term_document_frequencies[term] += 1
            else:
                self.term_document_frequencies[term] = 1

        self._calc_idf()
        self._calc_term_indices()

