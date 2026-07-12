from __future__ import annotations

from collections.abc import Sequence

from aise.contracts import (
    Query,
    SearchDocument,
    SearchResult,
    Retriever,
    VectorIndex
)


class DenseRetriever(Retriever):
    def __init__(
        self,
        index: VectorIndex,
        documents: Sequence[SearchDocument],
        model,
    ):
        # TODO:
        # Replace with the final embedding model and FAISS index
        # after the embeddings module is integrated.
        self.index = index
        self.documents = documents
        self.model = model

    def search(self, query: Query) -> Sequence[SearchResult]:
        # TODO:
        # Load the production embedding model and FAISS index
        # after the embeddings pipeline is completed.

        # get request's embedding
        query_vector = self.model.encode(
            [query.text],
            convert_to_numpy=True,
        )

        # looking for the nearests documents
        scores, indices = self.index.search(
            query_vector,
            query.top_k,
        )

        results: list[SearchResult] = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            # FAISS can return -1, if the number of results is less than top_k
            if idx < 0:
                continue

            doc = self.documents[idx]

            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    model_id=doc.model_id,
                    score=float(score),
                    rank=rank,
                    title=doc.title,
                    snippet=doc.body[:200],
                    metadata=doc.metadata,
                )
            )

        return results