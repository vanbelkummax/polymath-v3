"""
Secret management for Polymath v3.

Supports:
- Environment variables (.env for local dev)
- AWS Secrets Manager
- Google Cloud Secret Manager
- HashiCorp Vault

Usage:
    from lib.secrets import get_secret

    # Falls back through: ENV -> AWS -> GCP -> Vault -> None
    api_key = get_secret("GEMINI_API_KEY")

    # Or specify source explicitly
    password = get_secret("DB_PASSWORD", source="gcp")
"""

import logging
import os
from functools import lru_cache
from typing import Optional, Literal

logger = logging.getLogger(__name__)

SecretSource = Literal["env", "aws", "gcp", "vault", "auto"]


@lru_cache(maxsize=100)
def get_secret(
    key: str,
    source: SecretSource = "auto",
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Get a secret value.

    Args:
        key: Secret name/key
        source: Where to look for the secret:
            - "env": Environment variables only
            - "aws": AWS Secrets Manager
            - "gcp": Google Cloud Secret Manager
            - "vault": HashiCorp Vault
            - "auto": Try all sources in order
        default: Default value if not found

    Returns:
        Secret value or default
    """
    if source == "env":
        return os.environ.get(key, default)

    if source == "auto":
        # Try sources in order
        for src in ["env", "aws", "gcp", "vault"]:
            value = _get_from_source(key, src)
            if value is not None:
                return value
        return default

    return _get_from_source(key, source) or default


def _get_from_source(key: str, source: str) -> Optional[str]:
    """Get secret from specific source."""
    if source == "env":
        return os.environ.get(key)

    if source == "aws":
        return _get_from_aws(key)

    if source == "gcp":
        return _get_from_gcp(key)

    if source == "vault":
        return _get_from_vault(key)

    return None


def _get_from_aws(key: str) -> Optional[str]:
    """
    Get secret from AWS Secrets Manager.

    Requires:
        - boto3 installed
        - AWS credentials configured
        - AWS_REGION set

    Secret name format: polymath/{key}
    """
    try:
        import boto3
        import json

        client = boto3.client("secretsmanager")
        secret_name = f"polymath/{key}"

        response = client.get_secret_value(SecretId=secret_name)

        if "SecretString" in response:
            # May be JSON or plain string
            secret = response["SecretString"]
            try:
                data = json.loads(secret)
                return data.get(key) or data.get("value")
            except json.JSONDecodeError:
                return secret

        logger.debug(f"Secret {key} retrieved from AWS")
        return None

    except ImportError:
        logger.debug("boto3 not installed, skipping AWS")
        return None
    except Exception as e:
        logger.debug(f"AWS secret fetch failed for {key}: {e}")
        return None


def _get_from_gcp(key: str) -> Optional[str]:
    """
    Get secret from Google Cloud Secret Manager.

    Requires:
        - google-cloud-secret-manager installed
        - GCP credentials configured
        - GCP_PROJECT_ID set

    Secret name format: projects/{project}/secrets/{key}/versions/latest
    """
    try:
        from google.cloud import secretmanager

        project_id = os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            logger.debug("GCP_PROJECT_ID not set, skipping GCP secrets")
            return None

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{key}/versions/latest"

        response = client.access_secret_version(request={"name": name})
        secret = response.payload.data.decode("UTF-8")

        logger.debug(f"Secret {key} retrieved from GCP")
        return secret

    except ImportError:
        logger.debug("google-cloud-secret-manager not installed, skipping GCP")
        return None
    except Exception as e:
        logger.debug(f"GCP secret fetch failed for {key}: {e}")
        return None


def _get_from_vault(key: str) -> Optional[str]:
    """
    Get secret from HashiCorp Vault.

    Requires:
        - hvac installed
        - VAULT_ADDR set
        - VAULT_TOKEN or VAULT_ROLE_ID + VAULT_SECRET_ID set

    Secret path: secret/data/polymath/{key}
    """
    try:
        import hvac

        vault_addr = os.environ.get("VAULT_ADDR")
        if not vault_addr:
            logger.debug("VAULT_ADDR not set, skipping Vault")
            return None

        client = hvac.Client(url=vault_addr)

        # Try token auth first
        token = os.environ.get("VAULT_TOKEN")
        if token:
            client.token = token
        else:
            # Try AppRole auth
            role_id = os.environ.get("VAULT_ROLE_ID")
            secret_id = os.environ.get("VAULT_SECRET_ID")
            if role_id and secret_id:
                client.auth.approle.login(
                    role_id=role_id,
                    secret_id=secret_id,
                )

        if not client.is_authenticated():
            logger.debug("Vault authentication failed")
            return None

        secret_path = f"secret/data/polymath/{key}"
        response = client.secrets.kv.v2.read_secret_version(path=f"polymath/{key}")
        data = response["data"]["data"]

        logger.debug(f"Secret {key} retrieved from Vault")
        return data.get(key) or data.get("value")

    except ImportError:
        logger.debug("hvac not installed, skipping Vault")
        return None
    except Exception as e:
        logger.debug(f"Vault secret fetch failed for {key}: {e}")
        return None


def clear_cache():
    """Clear the secret cache (e.g., for rotation)."""
    get_secret.cache_clear()
    logger.info("Secret cache cleared")


def rotate_secrets(keys: list[str] = None) -> dict:
    """
    Force refresh of cached secrets.

    Args:
        keys: Specific keys to rotate, or None for all

    Returns:
        Dict of key -> success boolean
    """
    if keys is None:
        clear_cache()
        return {"all": True}

    results = {}
    for key in keys:
        # Clear from cache and refetch
        get_secret.cache_clear()
        value = get_secret(key)
        results[key] = value is not None

    return results


# Common secret names
SECRET_KEYS = [
    "POSTGRES_DSN",
    "NEO4J_PASSWORD",
    "GEMINI_API_KEY",
    "S2_API_KEY",
    "BRAVE_API_KEY",
    "GCP_SERVICE_ACCOUNT",
]
