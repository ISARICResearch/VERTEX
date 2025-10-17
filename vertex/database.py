import json
import os
from urllib.parse import quote_plus

import boto3


def get_database_url():
    env = os.getenv("APP_ENV", "local")
    if env == "ci":
        # GitHub Actions
        return "postgresql+psycopg2://test_user:test_password@localhost:5432/test_db"

    if env == "local":
        # Local development #TODO actually implement a sqlite dev environment
        return os.getenv("DATABASE_URL", "sqlite:///test.db")

    # Production mode – fetch from AWS Secrets Manager
    secret_name = os.getenv("AWS_SECRET_NAME")
    region_name = os.getenv("AWS_REGION", "eu-west-2")

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response["SecretString"])

    username = secret["username"]
    password = quote_plus(secret["password"])
    host = secret["host"]
    port = secret.get("port", 5432)
    database = secret["dbname"]

    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}?sslmode=require"
