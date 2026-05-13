import json
import os
from functools import lru_cache
from urllib.parse import quote_plus

import boto3

from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)


def get_database_url():
    env = os.getenv("APP_ENV", "local").strip().lower()
    if env == "ci":
        # GitHub Actions
        # TODO: seed test db for ci testing
        return "postgresql+psycopg2://test_user:test_password@localhost:5432/test_db"

    if env in ("local", "dev", "development"):
        # Local development #TODO actually implement a sqlite dev environment
        return os.getenv("DATABASE_URL", "sqlite:///test.db")

    # Production mode – fetch from AWS Secrets Manager
    secret_name = os.getenv("AWS_SECRET_NAME")
    region_name = os.getenv("AWS_REGION", "eu-west-2")

    if not secret_name:
        raise ValueError("AWS_SECRET_NAME must be set when APP_ENV is not local/dev/ci")

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response["SecretString"])

    username = secret["username"]
    password = quote_plus(secret["password"])
    host = os.getenv("DATABASE_HOST")
    port = os.getenv("DATABASE_PORT", 5432)
    database = "postgres"

    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}?sslmode=require"


@lru_cache(maxsize=4)
def get_aws_secret(secret_name: str, region_name: str) -> dict:
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def get_auth_settings():
    app_env = os.getenv("APP_ENV", "local").lower()
    region_name = os.getenv("AWS_REGION", "eu-west-2")

    if app_env in ("ci", "local", "dev", "development"):
        return {
            "SECRET_KEY": "local_dev_secret_key_mouse_trap",
            "COGNITO_CLIENT_ID": os.getenv("COGNITO_CLIENT_ID"),
            "COGNITO_CLIENT_SECRET": os.getenv("COGNITO_CLIENT_SECRET"),
            "COGNITO_USER_POOL_ID": os.getenv("COGNITO_USER_POOL_ID"),
            "COGNITO_DOMAIN": os.getenv("COGNITO_DOMAIN"),
            "VERTEX_BASE_URL": os.getenv("VERTEX_BASE_URL", "http://localhost:8050/auth"),
            "AWS_REGION": region_name,
        }

    secret_name = os.getenv("FLASK_AUTH_SECRETS")
    secret = get_aws_secret(secret_name, region_name)

    return {
        "SECRET_KEY": secret.get("SECRET_KEY"),
        "COGNITO_CLIENT_ID": secret.get("COGNITO_CLIENT_ID", os.getenv("COGNITO_CLIENT_ID")),
        "COGNITO_CLIENT_SECRET": secret.get("COGNITO_CLIENT_SECRET", os.getenv("COGNITO_CLIENT_SECRET")),
        "COGNITO_USER_POOL_ID": secret.get("COGNITO_USER_POOL_ID", os.getenv("COGNITO_USER_POOL_ID")),
        "COGNITO_DOMAIN": secret.get("COGNITO_DOMAIN", os.getenv("COGNITO_DOMAIN")),
        # Not a secret — always read from EC2 instance environment
        "VERTEX_BASE_URL": os.getenv("VERTEX_BASE_URL"),
        "AWS_REGION": secret.get("AWS_REGION", region_name),
    }


def get_flask_auth_secrets():
    return get_auth_settings()
