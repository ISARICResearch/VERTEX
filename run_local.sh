#!/bin/bash
set -euo pipefail

# === CONFIGURATION ===
AWS_PROFILE=$AWS_PROFILE_ISARIC # Your SSO profile name
ROLE_ARN=$ROLE_ARN_ISARIC # the isaric auth arn
SESSION_NAME="local-dev"
REGION="eu-west-2"
IMAGE_NAME="my-app-image"

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
echo "🐳 Launching Docker container..."

docker run \
  -v "$(pwd)":/app \
  -p 8050:8050 \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
  -e AWS_REGION="$REGION" \
  -w /app \
  -t vertex
