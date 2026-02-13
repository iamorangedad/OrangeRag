"""Dense retriever using vector similarity search."""

from typing import List, Any, Optional

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.core.embeddings import BaseEmbedding

from app.core.retrievers.base import BaseRetriever, NodeWithScore


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using vector embeddings and similarity search.

    This retriever uses vector embeddings to perform semantic similarity search.
    It supports any LlamaIndex embedding model and vector store.
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        vector_store: Optional[SimpleVectorStore] = None,
    ):
        """
        Initialize dense retriever.

        Args:
            embed_model: Embedding model for vectorization
            vector_store: Optional vector store (creates SimpleVectorStore if None)
        """
        self.embed_model = embed_model
        self.vector_store = vector_store or SimpleVectorStore()
        self._nodes: dict[str, TextNode] = {}

    def add_documents(self, nodes: List[TextNode]) -> None:
        """
        Add documents to the retriever.

        Args:
            nodes: List of text nodes to add
        """
        if not nodes:
            return

        # Generate embeddings for nodes
        texts = [node.text for node in nodes]
        embeddings = self.embed_model.get_text_embedding_batch(texts)

        # Set embeddings on nodes
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            self._nodes[node.id_] = node

        # Add to vector store
        self.vector_store.add(nodes)

    def retrieve(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Retrieve documents using vector similarity search.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of nodes with similarity scores
        """
        if self.is_empty:
            return []

        # Generate query embedding
        query_embedding = self.embed_model.get_text_embedding(query)

        # Perform similarity search
        query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
        result = self.vector_store.query(query_obj)

        # Build NodeWithScore results
        results = []
        for idx, node_id in enumerate(result.ids):
            if node_id in self._nodes:
                node = self._nodes[node_id]
                score = result.similarities[idx] if result.similarities else 0.0
                results.append(NodeWithScore(node=node, score=score))

        return results

    def clear(self) -> None:
        """Clear all documents from the retriever."""
        self.vector_store = SimpleVectorStore()
        self._nodes.clear()

    @property
    def is_empty(self) -> bool:
        """Check if retriever has no documents."""
        return len(self._nodes) == 0

    @property
    def document_count(self) -> int:
        """Get the number of documents."""
        return len(self._nodes)
