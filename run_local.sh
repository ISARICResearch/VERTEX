#!/bin/bash
# set -euo pipefail

# === Load local env vars ===
if [[ -f "$(dirname "$0")/.env" ]]; then
  set -a
  # shellcheck source=.env
  source "$(dirname "$0")/.env"
  set +a
fi

# === CONFIGURATION ===
# Prefer _ISARIC-suffixed vars from .env so they don't clobber the host AWS env
AWS_PROFILE="${AWS_PROFILE:-$AWS_PROFILE_ISARIC}"
ROLE_ARN="${ROLE_ARN:-$ROLE_ARN_ISARIC}"
SESSION_NAME="local-dev"
REGION="eu-west-2"
IMAGE_NAME="my-app-image"
PROJECTS_DIR="${VERTEX_PROJECTS_DIR:-$(pwd)/projects}"
USE_DOCKER="${USE_DOCKER:-true}"

# === Ensure you're logged in to SSO ===
echo "🔐 Ensuring SSO login is active..."
aws sso login --profile "$AWS_PROFILE"

# === Assume the target IAM role ===
echo "🔁 Assuming role: $ROLE_ARN ..."
CREDS_JSON=$(aws sts assume-role \
  --role-arn "$ROLE_ARN" \
  --role-session-name "$SESSION_NAME" \
  --profile "$AWS_PROFILE" \
  --output json)

AWS_ACCESS_KEY_ID=$(echo "$CREDS_JSON" | jq -r '.Credentials.AccessKeyId')
AWS_SECRET_ACCESS_KEY=$(echo "$CREDS_JSON" | jq -r '.Credentials.SecretAccessKey')
AWS_SESSION_TOKEN=$(echo "$CREDS_JSON" | jq -r '.Credentials.SessionToken')


# === Run the Docker container ===
echo "Launching Docker container..."
docker run \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)":/app \
  -v "$(pwd)/demo-projects:/app/demo-projects" \
  -v "$(pwd)/projects:/app/projects" \
  -p 8050:8050 \
  -e APP_ENV="dev" \
  -e VERTEX_GIT_SHA="$(git rev-parse HEAD)" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
  -e AWS_REGION="$REGION" \
  -e AWS_SECRET_NAME="$AWS_SECRET_NAME" \
  -e DATABASE_URL="$DATABASE_URL" \
  -e COGNITO_CLIENT_ID="$COGNITO_CLIENT_ID" \
  -e COGNITO_CLIENT_SECRET="$COGNITO_CLIENT_SECRET" \
  -e COGNITO_USER_POOL_ID="$COGNITO_USER_POOL_ID" \
  -e COGNITO_DOMAIN="$COGNITO_DOMAIN" \
  -e VERTEX_BASE_URL="http://localhost:8050/auth" \
  -e FLASK_AUTH_SECRETS="arn:aws:secretsmanager:eu-west-2:891612573996:secret:isaric/flask-auth-secrets-5Iz1xn" \
  -e VERTEX_ENABLE_AUTH="True" \
  -e DATABASE_HOST="$DATABASE_HOST" \
  -e DATABASE_PORT="$DATABASE_PORT" \
  -e VERTEX_PROJECTS_DIR="/app/projects" \
  -e VERTEX_ENABLE_SAVE_OUTPUTS="true" \
  -w /app \
  -t vertex
