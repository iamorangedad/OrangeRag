"""Query expansion and rewriting module for improving retrieval."""

import re
from typing import List, Optional
from abc import ABC, abstractmethod


class QueryExpander(ABC):
    """Abstract base class for query expanders."""

    @abstractmethod
    def expand(self, query: str) -> List[str]:
        """
        Expand a query into multiple variations.

        Args:
            query: Original query string

        Returns:
            List of expanded queries (including original)
        """
        pass


class SynonymExpander(QueryExpander):
    """
    Query expander using synonym replacement.

    This expander generates variations by replacing words with common synonyms.
    Uses a simple built-in synonym dictionary.
    """

    # Simple synonym dictionary
    SYNONYMS = {
        "how to": ["how do I", "what is the way to", "steps to"],
        "create": ["make", "build", "generate"],
        "find": ["locate", "search for", "discover"],
        "use": ["utilize", "apply", "employ"],
        "increase": ["boost", "enhance", "improve", "raise"],
        "decrease": ["reduce", "lower", "diminish", "cut"],
        "best": ["optimal", "top", "finest", "most suitable"],
        "worst": ["worst case", "least favorable", "poorest"],
        "example": ["instance", "sample", "illustration"],
        "problem": ["issue", "challenge", "difficulty", "obstacle"],
        "solution": ["answer", "resolution", "fix", "remedy"],
        "method": ["approach", "technique", "strategy", "way"],
        "important": ["crucial", "essential", "vital", "significant"],
        "fast": ["quick", "rapid", "swift", "speedy"],
        "slow": ["sluggish", "gradual", "unhurried"],
        "large": ["big", "huge", "enormous", "substantial"],
        "small": ["tiny", "little", "minor", "compact"],
        "easy": ["simple", "straightforward", "effortless"],
        "difficult": ["hard", "challenging", "complex", "complicated"],
        "beginner": ["novice", "newcomer", "starter"],
        "expert": ["professional", "specialist", "authority"],
    }

    def __init__(self, max_expansions: int = 3):
        """
        Initialize synonym expander.

        Args:
            max_expansions: Maximum number of expanded queries to generate
        """
        self.max_expansions = max_expansions

    def expand(self, query: str) -> List[str]:
        """
        Expand query using synonym replacement.

        Args:
            query: Original query string

        Returns:
            List of expanded queries including original
        """
        query_lower = query.lower()
        expansions = [query]  # Always include original

        # Find synonyms for words in query
        for word, synonyms in self.SYNONYMS.items():
            if word in query_lower:
                for synonym in synonyms[: self.max_expansions]:
                    expanded = re.sub(
                        r"\b" + re.escape(word) + r"\b", synonym, query_lower, flags=re.IGNORECASE
                    )
                    if expanded != query_lower and expanded not in expansions:
                        expansions.append(expanded)

        return expansions[: self.max_expansions + 1]


class KeywordExpander(QueryExpander):
    """
    Query expander using keyword extraction and expansion.

    Extracts key terms and generates variations with additional context.
    """

    def __init__(self, add_context_terms: bool = True):
        """
        Initialize keyword expander.

        Args:
            add_context_terms: Whether to add contextual terms
        """
        self.add_context_terms = add_context_terms

    def expand(self, query: str) -> List[str]:
        """
        Expand query by adding context terms.

        Args:
            query: Original query string

        Returns:
            List of expanded queries
        """
        expansions = [query]

        # Add question variations if it's a question
        if self._is_question(query):
            # Add "what is" prefix if not present
            if not query.lower().startswith(
                ("what", "how", "why", "when", "where", "who", "which")
            ):
                expansions.append(f"what is {query}")
                expansions.append(f"how to {query}")

        # Add formal/informal variations
        if "how do I" in query.lower():
            expansions.append(query.lower().replace("how do I", "how to"))
        elif "how to" in query.lower():
            expansions.append(query.lower().replace("how to", "how do I"))

        return list(set(expansions))  # Remove duplicates

    def _is_question(self, query: str) -> bool:
        """Check if query is a question."""
        question_starters = (
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "is",
            "are",
            "can",
            "does",
            "do",
        )
        return query.lower().startswith(question_starters) or query.endswith("?")


class HyDEExpander(QueryExpander):
    """
    HyDE (Hypothetical Document Embeddings) style expander.

    This expander uses an LLM to generate hypothetical documents that
    would answer the query, then uses those as expanded queries.
    """

    def __init__(self, llm=None, num_hypotheses: int = 3):
        """
        Initialize HyDE expander.

        Args:
            llm: LLM instance for generating hypothetical documents
            num_hypotheses: Number of hypothetical documents to generate
        """
        self.llm = llm
        self.num_hypotheses = num_hypotheses

    def expand(self, query: str) -> List[str]:
        """
        Expand query using HyDE approach.

        Args:
            query: Original query string

        Returns:
            List of expanded queries (including hypothetical documents)
        """
        if self.llm is None:
            return [query]

        expansions = [query]

        try:
            # Generate hypothetical document
            prompt = f"""Generate a short paragraph that would be a good answer to this query.
Query: {query}

Answer:"""

            response = self.llm.complete(prompt)
            hypothetical_doc = str(response).strip()

            if hypothetical_doc and len(hypothetical_doc) > 20:
                expansions.append(hypothetical_doc)

        except Exception as e:
            # If LLM fails, just return original query
            pass

        return expansions


class MultiExpander(QueryExpander):
    """
    Composite expander that chains multiple expansion strategies.
    """

    def __init__(self, expanders: Optional[List[QueryExpander]] = None):
        """
        Initialize multi-expander.

        Args:
            expanders: List of expanders to chain (default: synonym + keyword)
        """
        if expanders is None:
            expanders = [SynonymExpander(), KeywordExpander()]
        self.expanders = expanders

    def expand(self, query: str) -> List[str]:
        """
        Expand query using all configured expanders.

        Args:
            query: Original query string

        Returns:
            Combined list of expanded queries
        """
        all_expansions = set([query])

        for expander in self.expanders:
            try:
                expansions = expander.expand(query)
                all_expansions.update(expansions)
            except Exception:
                # Skip expander if it fails
                continue

        return list(all_expansions)


class QueryRewriter:
    """
    Query rewriter for transforming queries into better retrieval formats.
    """

    @staticmethod
    def rewrite_for_retrieval(query: str) -> str:
        """
        Rewrite query to be more suitable for retrieval.

        Args:
            query: Original query

        Returns:
            Rewritten query
        """
        rewritten = query.strip()

        # Remove question marks for better keyword matching
        if rewritten.endswith("?"):
            rewritten = rewritten[:-1].strip()

        # Convert to lowercase for consistency
        rewritten = rewritten.lower()

        # Remove common conversational prefixes
        prefixes_to_remove = [
            "can you tell me",
            "could you explain",
            "i want to know",
            "i would like to know",
            "please tell me",
            "do you know",
        ]

        for prefix in prefixes_to_remove:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix) :].strip()
                # Remove leading "about" or "how"
                if rewritten.startswith(("about ", "how ")):
                    rewritten = rewritten[6:].strip()

        return rewritten

    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """
        Extract important keywords from query.

        Args:
            query: Query string

        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "and",
            "but",
            "or",
            "yet",
            "so",
            "for",
            "nor",
            "in",
            "on",
            "at",
            "to",
            "from",
            "by",
            "with",
            "about",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
        }

        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords
