#!/usr/bin/env bash
set -euo pipefail

SYNC_SCRIPT_PATH="/usr/local/bin/vertex-sync-projects.sh"
CRON_FILE_PATH="/etc/cron.d/vertex-projects-sync"
CRON_LOG_PATH="/var/log/vertex-projects-sync.log"

cat > "${CRON_FILE_PATH}" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
PROJECTS_REPO_URL=git@github.com:ISARICResearch/VERTEX-projects.git
PROJECTS_REPO_BRANCH=main
VERTEX_PROJECTS_DIR=/opt/vertex-projects
PROJECTS_GIT_SSH_KEY_PATH=/root/.ssh/vertex_projects_deploy_key
PROJECTS_GIT_KNOWN_HOSTS_PATH=/root/.ssh/known_hosts
VERTEX_SYNC_CONTAINER_NAME=isaric-vertex
0 * * * * root { ts=$(date -u --iso-8601=seconds); echo "[${ts}] vertex-projects cron run started"; ${SYNC_SCRIPT_PATH}; sync_rc=$?; if [ ${sync_rc} -ne 0 ]; then ts=$(date -u --iso-8601=seconds); echo "[${ts}] sync failed rc=${sync_rc}"; exit ${sync_rc}; fi; ts=$(date -u --iso-8601=seconds); echo "[${ts}] ingestion started container=${VERTEX_SYNC_CONTAINER_NAME} projects_dir=${VERTEX_PROJECTS_DIR}"; docker exec ${VERTEX_SYNC_CONTAINER_NAME} python -m vertex.project_ingestion --projects-dir ${VERTEX_PROJECTS_DIR}; ingest_rc=$?; ts=$(date -u --iso-8601=seconds); echo "[${ts}] vertex-projects cron run finished rc=${ingest_rc}"; exit ${ingest_rc}; } >> ${CRON_LOG_PATH} 2>&1
EOF

chmod 644 "${CRON_FILE_PATH}"
echo "Installed hourly projects sync cron at ${CRON_FILE_PATH}"
