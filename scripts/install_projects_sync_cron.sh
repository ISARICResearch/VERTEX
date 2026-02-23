#!/usr/bin/env bash
set -euo pipefail

SYNC_SCRIPT_PATH="${SYNC_SCRIPT_PATH:-/usr/local/bin/vertex-sync-projects.sh}"
CRON_FILE_PATH="${CRON_FILE_PATH:-/etc/cron.d/vertex-projects-sync}"
CRON_LOG_PATH="${CRON_LOG_PATH:-/var/log/vertex-projects-sync.log}"

cat > "${CRON_FILE_PATH}" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
PROJECTS_REPO_URL=git@github.com:ISARICResearch/VERTEX-projects.git
PROJECTS_REPO_BRANCH=${PROJECTS_REPO_BRANCH:-main}
VERTEX_PROJECTS_DIR=${VERTEX_PROJECTS_DIR:-/opt/vertex-projects}
PROJECTS_GIT_SSH_KEY_PATH=${PROJECTS_GIT_SSH_KEY_PATH:-/root/.ssh/vertex_projects_deploy_key}
PROJECTS_GIT_KNOWN_HOSTS_PATH=${PROJECTS_GIT_KNOWN_HOSTS_PATH:-/root/.ssh/known_hosts}
0 * * * * root ${SYNC_SCRIPT_PATH} >> ${CRON_LOG_PATH} 2>&1
EOF

chmod 644 "${CRON_FILE_PATH}"
echo "Installed hourly projects sync cron at ${CRON_FILE_PATH}"
