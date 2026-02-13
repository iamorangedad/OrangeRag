"""Unit tests for query expansion module."""

import pytest
from unittest.mock import Mock

from app.core.query.expansion import (
    SynonymExpander,
    KeywordExpander,
    HyDEExpander,
    MultiExpander,
    QueryRewriter,
)


class TestSynonymExpander:
    """Test synonym-based query expansion."""

    def test_initialization(self):
        """Test expander initialization."""
        expander = SynonymExpander(max_expansions=5)
        assert expander.max_expansions == 5

    def test_expand_returns_original(self):
        """Test that expansion always returns original query."""
        expander = SynonymExpander()
        query = "test query"
        expansions = expander.expand(query)

        assert query in expansions

    def test_expand_with_synonyms(self):
        """Test expansion with synonym replacement."""
        expander = SynonymExpander(max_expansions=3)
        query = "how to create a method"
        expansions = expander.expand(query)

        assert len(expansions) > 1
        assert query in expansions

    def test_expand_respects_max_expansions(self):
        """Test that max_expansions is respected."""
        expander = SynonymExpander(max_expansions=2)
        query = "how to use the best example"
        expansions = expander.expand(query)

        assert len(expansions) <= 3  # Original + max_expansions

    def test_expand_no_synonyms_found(self):
        """Test expansion when no synonyms match."""
        expander = SynonymExpander()
        query = "xyz abc unknown words"
        expansions = expander.expand(query)

        assert expansions == [query]

    def test_expand_case_insensitive(self):
        """Test that expansion is case insensitive."""
        expander = SynonymExpander()
        query = "How To Create"
        expansions = expander.expand(query)

        assert len(expansions) > 1


class TestKeywordExpander:
    """Test keyword-based query expansion."""

    def test_initialization(self):
        """Test expander initialization."""
        expander = KeywordExpander(add_context_terms=True)
        assert expander.add_context_terms is True

    def test_expand_returns_original(self):
        """Test that expansion returns original query."""
        expander = KeywordExpander()
        query = "test query"
        expansions = expander.expand(query)

        assert query in expansions

    def test_expand_question_variations(self):
        """Test expansion adds variations for questions."""
        expander = KeywordExpander()
        query = "install Python"
        expansions = expander.expand(query)

        assert len(expansions) > 1

    def test_is_question_with_question_mark(self):
        """Test question detection with question mark."""
        expander = KeywordExpander()
        assert expander._is_question("what is this?") is True
        assert expander._is_question("this is a statement") is False

    def test_is_question_with_starters(self):
        """Test question detection with question starters."""
        expander = KeywordExpander()
        assert expander._is_question("How do I use this") is True
        assert expander._is_question("What is Python") is True
        assert expander._is_question("Can I do this") is True

    def test_how_do_i_variation(self):
        """Test how do I / how to variations."""
        expander = KeywordExpander()
        query1 = "how do I create a function"
        query2 = "how to create a function"

        expansions1 = expander.expand(query1)
        expansions2 = expander.expand(query2)

        # Check that variations are added
        assert any("how to" in e for e in expansions1)
        assert any("how do I" in e for e in expansions2)

    def test_no_duplicate_expansions(self):
        """Test that duplicates are removed."""
        expander = KeywordExpander()
        query = "how to use Python"
        expansions = expander.expand(query)

        assert len(expansions) == len(set(expansions))


class TestHyDEExpander:
    """Test HyDE query expansion."""

    def test_initialization_without_llm(self):
        """Test initialization without LLM."""
        expander = HyDEExpander(llm=None)
        assert expander.llm is None
        assert expander.num_hypotheses == 3

    def test_expand_without_llm(self):
        """Test expansion without LLM returns original."""
        expander = HyDEExpander(llm=None)
        query = "test query"
        expansions = expander.expand(query)

        assert expansions == [query]

    def test_expand_with_llm(self):
        """Test expansion with mock LLM."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock()
        mock_llm.complete.return_value.__str__ = Mock(return_value="This is a hypothetical answer.")

        expander = HyDEExpander(llm=mock_llm)
        query = "test query"
        expansions = expander.expand(query)

        assert query in expansions
        assert len(expansions) == 2
        mock_llm.complete.assert_called_once()

    def test_expand_with_short_response(self):
        """Test that short LLM responses are ignored."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock()
        mock_llm.complete.return_value.__str__ = Mock(return_value="Short.")

        expander = HyDEExpander(llm=mock_llm)
        query = "test query"
        expansions = expander.expand(query)

        assert expansions == [query]

    def test_expand_with_llm_error(self):
        """Test expansion handles LLM errors gracefully."""
        mock_llm = Mock()
        mock_llm.complete.side_effect = Exception("LLM error")

        expander = HyDEExpander(llm=mock_llm)
        query = "test query"
        expansions = expander.expand(query)

        assert expansions == [query]


class TestMultiExpander:
    """Test multi-expander that combines strategies."""

    def test_initialization_default(self):
        """Test initialization with default expanders."""
        expander = MultiExpander()
        assert len(expander.expanders) == 2
        assert isinstance(expander.expanders[0], SynonymExpander)
        assert isinstance(expander.expanders[1], KeywordExpander)

    def test_initialization_custom(self):
        """Test initialization with custom expanders."""
        custom_expanders = [SynonymExpander(), SynonymExpander()]
        expander = MultiExpander(expanders=custom_expanders)
        assert expander.expanders == custom_expanders

    def test_expand_combines_results(self):
        """Test that expansion combines results from all expanders."""
        expander = MultiExpander()
        query = "how to create the best example"
        expansions = expander.expand(query)

        assert query in expansions
        assert len(expansions) > 1

    def test_expand_removes_duplicates(self):
        """Test that expansion removes duplicates."""
        expander = MultiExpander()
        query = "test"
        expansions = expander.expand(query)

        assert len(expansions) == len(set(expansions))

    def test_expand_handles_expander_errors(self):
        """Test that expansion handles expander errors gracefully."""
        failing_expander = Mock()
        failing_expander.expand.side_effect = Exception("Expander error")

        working_expander = SynonymExpander()

        expander = MultiExpander(expanders=[failing_expander, working_expander])
        query = "test"
        expansions = expander.expand(query)

        assert query in expansions


class TestQueryRewriter:
    """Test query rewriting utilities."""

    def test_rewrite_removes_question_mark(self):
        """Test that question marks are removed."""
        query = "what is python?"
        rewritten = QueryRewriter.rewrite_for_retrieval(query)

        assert "?" not in rewritten

    def test_rewrite_converts_to_lowercase(self):
        """Test conversion to lowercase."""
        query = "What Is Python"
        rewritten = QueryRewriter.rewrite_for_retrieval(query)

        assert rewritten == rewritten.lower()

    def test_rewrite_removes_prefixes(self):
        """Test removal of conversational prefixes."""
        query = "can you tell me about python"
        rewritten = QueryRewriter.rewrite_for_retrieval(query)

        assert "can you tell me" not in rewritten

    def test_rewrite_removes_about_after_prefix(self):
        """Test removal of 'about' after prefix removal."""
        query = "i want to know about python"
        rewritten = QueryRewriter.rewrite_for_retrieval(query)

        assert not rewritten.startswith("about")

    def test_rewrite_handles_simple_query(self):
        """Test rewrite with simple query."""
        query = "python tutorial"
        rewritten = QueryRewriter.rewrite_for_retrieval(query)

        assert rewritten == "python tutorial"

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        query = "the quick brown fox"
        keywords = QueryRewriter.extract_keywords(query)

        assert "the" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_extract_keywords_removes_short_words(self):
        """Test that short words are removed."""
        query = "how to use python"
        keywords = QueryRewriter.extract_keywords(query)

        # Words with length <= 2 should be removed
        assert all(len(k) > 2 for k in keywords)

    def test_extract_keywords_lowercase(self):
        """Test that keywords are lowercased."""
        query = "Python Programming"
        keywords = QueryRewriter.extract_keywords(query)

        assert all(k == k.lower() for k in keywords)

    def test_extract_keywords_empty(self):
        """Test keyword extraction with empty query."""
        query = ""
        keywords = QueryRewriter.extract_keywords(query)

        assert keywords == []

    def test_extract_keywords_all_stop_words(self):
        """Test extraction when all words are stop words."""
        query = "the and or but"
        keywords = QueryRewriter.extract_keywords(query)

        assert keywords == []
