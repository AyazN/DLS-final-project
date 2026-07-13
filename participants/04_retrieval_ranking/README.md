# 04 Retrieval + Ranking

This directory contains the team's retrieval and ranking implementation:

- tokenizer.py provides tokenization;
- bm25.py provides BM25Retriever;
- dense.py provides DenseRetriever;
- rrf.py provides ReciprocalRankFusion;
- hybrid.py provides HybridRetriever;
- reranker.py provides CrossEncoderReranker.

Retriever classes implement the shared Retriever.search(Query) contract. CrossEncoderReranker implements Reranker.rank(Query, candidates). Every stage returns a sequence of SearchResult objects.
