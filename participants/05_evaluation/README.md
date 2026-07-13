# 05 Evaluation + Metrics

RetrievalEvaluator computes Precision@K, Recall@K, MRR, and nDCG@K.

The evaluator uses the shared contracts:

- it calls Retriever.search(Query);
- it compares SearchResult.model_id with EvaluationExample.relevant_model_ids;
- it returns an EvaluationReport;
- it can save a JSON report under data/results.
