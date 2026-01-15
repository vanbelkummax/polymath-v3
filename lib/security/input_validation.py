"""
Input validation utilities for Polymath v3.

Provides validation and sanitization for user inputs across the system.
"""

import re
from pathlib import Path
from typing import Optional


class InputValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def validate_search_query(
    query: str,
    max_length: int = 1000,
    min_length: int = 1,
) -> str:
    """
    Validate and sanitize a search query.

    Args:
        query: The search query to validate
        max_length: Maximum allowed query length
        min_length: Minimum allowed query length

    Returns:
        Sanitized query string

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(query, str):
        raise InputValidationError("Query must be a string")

    # Remove leading/trailing whitespace
    query = query.strip()

    if len(query) < min_length:
        raise InputValidationError(f"Query too short (min {min_length} characters)")

    if len(query) > max_length:
        raise InputValidationError(f"Query too long (max {max_length} characters)")

    # Remove control characters (keep newlines and tabs)
    query = "".join(c for c in query if ord(c) >= 32 or c in "\n\t")

    return query


def validate_uuid(value: str) -> str:
    """
    Validate UUID format.

    Args:
        value: String to validate as UUID

    Returns:
        Normalized UUID (lowercase)

    Raises:
        InputValidationError: If not a valid UUID format
    """
    if not isinstance(value, str):
        raise InputValidationError("UUID must be a string")

    # Standard UUID format: 8-4-4-4-12 hex digits
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

    if not re.match(uuid_pattern, value, re.IGNORECASE):
        raise InputValidationError(f"Invalid UUID format: {value}")

    return value.lower()


def validate_file_path(
    path: str | Path,
    allowed_extensions: Optional[list[str]] = None,
    allowed_dirs: Optional[list[Path]] = None,
    max_size_mb: int = 500,
    must_exist: bool = True,
) -> Path:
    """
    Validate and sanitize a file path.

    Checks:
    - Path resolves to a valid canonical path
    - Extension is allowed (if specified)
    - Path is within allowed directories (if specified)
    - File exists and is within size limits (if must_exist=True)

    Args:
        path: File path to validate
        allowed_extensions: List of allowed extensions (e.g., [".pdf", ".txt"])
        allowed_dirs: List of allowed parent directories
        max_size_mb: Maximum file size in MB
        must_exist: Whether file must exist

    Returns:
        Resolved Path object

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(path, (str, Path)):
        raise InputValidationError("Path must be a string or Path object")

    # Convert to Path and resolve
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError) as e:
        raise InputValidationError(f"Invalid path: {e}")

    # Check extension
    if allowed_extensions:
        if resolved.suffix.lower() not in allowed_extensions:
            raise InputValidationError(
                f"Invalid file type: {resolved.suffix}. Allowed: {allowed_extensions}"
            )

    # Check if within allowed directories
    if allowed_dirs:
        in_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                resolved.relative_to(allowed_dir.resolve())
                in_allowed = True
                break
            except ValueError:
                continue

        if not in_allowed:
            raise InputValidationError(
                f"Path is outside allowed directories: {resolved}"
            )

    # Check existence and size
    if must_exist:
        if not resolved.exists():
            raise InputValidationError(f"File not found: {resolved}")

        if not resolved.is_file():
            raise InputValidationError(f"Not a file: {resolved}")

        try:
            size_mb = resolved.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise InputValidationError(
                    f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
                )
        except OSError as e:
            raise InputValidationError(f"Cannot access file: {e}")

    return resolved


def sanitize_string(
    value: str,
    max_length: Optional[int] = None,
    allow_newlines: bool = True,
    allow_tabs: bool = True,
) -> str:
    """
    Sanitize a string by removing control characters.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length (truncate if exceeded)
        allow_newlines: Whether to keep newline characters
        allow_tabs: Whether to keep tab characters

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)

    # Build set of allowed control characters
    allowed_controls = set()
    if allow_newlines:
        allowed_controls.update({"\n", "\r"})
    if allow_tabs:
        allowed_controls.add("\t")

    # Remove control characters except allowed ones
    result = "".join(
        c for c in value if ord(c) >= 32 or c in allowed_controls
    )

    # Truncate if needed
    if max_length and len(result) > max_length:
        result = result[:max_length]

    return result


def validate_integer_range(
    value: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    name: str = "value",
) -> int:
    """
    Validate an integer is within a range.

    Args:
        value: Integer to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Returns:
        The validated integer

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise InputValidationError(f"{name} must be an integer")

    if min_val is not None and value < min_val:
        raise InputValidationError(f"{name} must be >= {min_val}")

    if max_val is not None and value > max_val:
        raise InputValidationError(f"{name} must be <= {max_val}")

    return value
