"""
Tests for security validation functions.
"""

import pytest
import tempfile
from pathlib import Path

from lib.security.input_validation import (
    validate_search_query,
    validate_uuid,
    validate_file_path,
    sanitize_string,
    validate_integer_range,
    InputValidationError,
)

# Import MCP server validation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp"))
from polymath_server import validate_cypher_read_only


class TestCypherValidation:
    """Tests for Cypher query security validation."""

    def test_valid_match_query(self):
        """MATCH queries should be allowed."""
        is_valid, error = validate_cypher_read_only("MATCH (n) RETURN n")
        assert is_valid is True
        assert error == ""

    def test_valid_complex_match(self):
        """Complex MATCH queries should be allowed."""
        query = """
        MATCH (p:Passage)-[:MENTIONS]->(m:METHOD)
        WHERE m.name CONTAINS 'transformer'
        RETURN p.passage_id, m.name
        LIMIT 10
        """
        is_valid, error = validate_cypher_read_only(query)
        assert is_valid is True

    def test_rejects_delete(self):
        """DELETE queries should be rejected."""
        is_valid, error = validate_cypher_read_only("MATCH (n) DELETE n")
        assert is_valid is False
        assert "DELETE" in error

    def test_rejects_detach_delete(self):
        """DETACH DELETE should be rejected."""
        is_valid, error = validate_cypher_read_only("MATCH (n) DETACH DELETE n")
        assert is_valid is False
        assert "DETACH DELETE" in error or "DELETE" in error

    def test_rejects_create(self):
        """CREATE queries should be rejected."""
        is_valid, error = validate_cypher_read_only("CREATE (n:Test {name: 'test'})")
        assert is_valid is False
        assert "CREATE" in error

    def test_rejects_merge(self):
        """MERGE queries should be rejected."""
        is_valid, error = validate_cypher_read_only("MERGE (n:Test {name: 'test'})")
        assert is_valid is False
        assert "MERGE" in error

    def test_rejects_set(self):
        """SET queries should be rejected."""
        is_valid, error = validate_cypher_read_only("MATCH (n) SET n.name = 'test'")
        assert is_valid is False
        assert "SET" in error

    def test_rejects_remove(self):
        """REMOVE queries should be rejected."""
        is_valid, error = validate_cypher_read_only("MATCH (n) REMOVE n.name")
        assert is_valid is False
        assert "REMOVE" in error

    def test_rejects_drop(self):
        """DROP queries should be rejected."""
        is_valid, error = validate_cypher_read_only("DROP INDEX idx_name")
        assert is_valid is False
        assert "DROP" in error

    def test_rejects_dbms_calls(self):
        """CALL dbms.* should be rejected."""
        is_valid, error = validate_cypher_read_only("CALL dbms.cluster.overview()")
        assert is_valid is False
        assert "CALL DBMS" in error

    def test_rejects_apoc_load(self):
        """CALL apoc.load.* should be rejected."""
        is_valid, error = validate_cypher_read_only("CALL apoc.load.json('http://evil.com')")
        assert is_valid is False
        assert "APOC.LOAD" in error

    def test_case_insensitive(self):
        """Validation should be case-insensitive."""
        is_valid, error = validate_cypher_read_only("match (n) delete n")
        assert is_valid is False
        assert "DELETE" in error


class TestSearchQueryValidation:
    """Tests for search query validation."""

    def test_valid_query(self):
        """Valid queries should pass."""
        result = validate_search_query("spatial transcriptomics")
        assert result == "spatial transcriptomics"

    def test_trims_whitespace(self):
        """Whitespace should be trimmed."""
        result = validate_search_query("  query with spaces  ")
        assert result == "query with spaces"

    def test_rejects_empty(self):
        """Empty queries should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_search_query("")
        assert "too short" in str(exc_info.value)

    def test_rejects_whitespace_only(self):
        """Whitespace-only queries should be rejected."""
        with pytest.raises(InputValidationError):
            validate_search_query("   ")

    def test_rejects_too_long(self):
        """Queries exceeding max_length should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_search_query("a" * 1001)
        assert "too long" in str(exc_info.value)

    def test_custom_max_length(self):
        """Custom max_length should be respected."""
        with pytest.raises(InputValidationError):
            validate_search_query("toolong", max_length=5)

    def test_removes_control_chars(self):
        """Control characters should be removed."""
        result = validate_search_query("query\x00with\x01control\x02chars")
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "querywithcontrolchars"

    def test_preserves_newlines(self):
        """Newlines should be preserved."""
        result = validate_search_query("line1\nline2")
        assert result == "line1\nline2"

    def test_rejects_non_string(self):
        """Non-string input should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_search_query(123)
        assert "must be a string" in str(exc_info.value)


class TestUUIDValidation:
    """Tests for UUID validation."""

    def test_valid_uuid(self):
        """Valid UUIDs should pass."""
        result = validate_uuid("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_uppercase_uuid(self):
        """Uppercase UUIDs should be normalized to lowercase."""
        result = validate_uuid("550E8400-E29B-41D4-A716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_rejects_invalid_format(self):
        """Invalid UUID formats should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_uuid("not-a-uuid")
        assert "Invalid UUID" in str(exc_info.value)

    def test_rejects_too_short(self):
        """Short strings should be rejected."""
        with pytest.raises(InputValidationError):
            validate_uuid("550e8400-e29b")

    def test_rejects_wrong_characters(self):
        """Non-hex characters should be rejected."""
        with pytest.raises(InputValidationError):
            validate_uuid("550g8400-e29b-41d4-a716-446655440000")

    def test_rejects_non_string(self):
        """Non-string input should be rejected."""
        with pytest.raises(InputValidationError):
            validate_uuid(123)


class TestFilePathValidation:
    """Tests for file path validation."""

    def test_valid_file(self):
        """Valid file paths should pass."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            result = validate_file_path(temp_path, allowed_extensions=[".pdf"])
            assert result == temp_path.resolve()
        finally:
            temp_path.unlink()

    def test_rejects_missing_file(self):
        """Missing files should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_file_path("/nonexistent/file.pdf")
        assert "not found" in str(exc_info.value).lower()

    def test_rejects_wrong_extension(self):
        """Wrong extensions should be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(InputValidationError) as exc_info:
                validate_file_path(temp_path, allowed_extensions=[".pdf"])
            assert "Invalid file type" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_rejects_too_large(self):
        """Files exceeding size limit should be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            # Write 2MB of data
            f.write(b"x" * (2 * 1024 * 1024))
            temp_path = Path(f.name)

        try:
            with pytest.raises(InputValidationError) as exc_info:
                validate_file_path(temp_path, max_size_mb=1)
            assert "too large" in str(exc_info.value).lower()
        finally:
            temp_path.unlink()

    def test_optional_existence_check(self):
        """must_exist=False should skip existence check."""
        result = validate_file_path(
            "/nonexistent/file.pdf",
            allowed_extensions=[".pdf"],
            must_exist=False,
        )
        assert result == Path("/nonexistent/file.pdf").resolve()


class TestSanitizeString:
    """Tests for string sanitization."""

    def test_normal_string(self):
        """Normal strings should pass through unchanged."""
        result = sanitize_string("Hello, world!")
        assert result == "Hello, world!"

    def test_removes_control_chars(self):
        """Control characters should be removed."""
        result = sanitize_string("Hello\x00World")
        assert result == "HelloWorld"

    def test_preserves_newlines(self):
        """Newlines should be preserved by default."""
        result = sanitize_string("line1\nline2")
        assert result == "line1\nline2"

    def test_removes_newlines_when_disabled(self):
        """Newlines should be removed when disabled."""
        result = sanitize_string("line1\nline2", allow_newlines=False)
        assert result == "line1line2"

    def test_preserves_tabs(self):
        """Tabs should be preserved by default."""
        result = sanitize_string("col1\tcol2")
        assert result == "col1\tcol2"

    def test_truncates_to_max_length(self):
        """Long strings should be truncated."""
        result = sanitize_string("a" * 100, max_length=10)
        assert len(result) == 10

    def test_non_string_converted(self):
        """Non-strings should be converted."""
        result = sanitize_string(123)
        assert result == "123"


class TestIntegerRangeValidation:
    """Tests for integer range validation."""

    def test_valid_value(self):
        """Valid values should pass."""
        result = validate_integer_range(5, min_val=0, max_val=10)
        assert result == 5

    def test_rejects_below_min(self):
        """Values below min should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_integer_range(-1, min_val=0)
        assert ">= 0" in str(exc_info.value)

    def test_rejects_above_max(self):
        """Values above max should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_integer_range(11, max_val=10)
        assert "<= 10" in str(exc_info.value)

    def test_rejects_non_integer(self):
        """Non-integers should be rejected."""
        with pytest.raises(InputValidationError) as exc_info:
            validate_integer_range("5", min_val=0)
        assert "must be an integer" in str(exc_info.value)

    def test_boundary_values(self):
        """Boundary values should be accepted."""
        assert validate_integer_range(0, min_val=0, max_val=10) == 0
        assert validate_integer_range(10, min_val=0, max_val=10) == 10
