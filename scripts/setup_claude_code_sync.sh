#!/bin/bash
set -e

# Setup macOS LaunchAgent for EngramVault Claude Code Sync
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
REMOTE_URL="${1:-http://localhost:5006}"
PLIST_SRC="${SCRIPT_DIR}/net.gaskb.engramvault.claude-code-sync.plist"
PLIST_DEST="${HOME}/Library/LaunchAgents/net.gaskb.engramvault.claude-code-sync.plist"

echo "=== Configurazione LaunchAgent EngramVault Claude Code Sync ==="
echo "Project Dir : ${PROJECT_DIR}"
echo "Python Bin  : ${PYTHON_BIN}"
echo "Remote URL  : ${REMOTE_URL}"
echo "Target Plist: ${PLIST_DEST}"

if [ ! -f "${PYTHON_BIN}" ]; then
    echo "Errore: Python bin non trovato in ${PYTHON_BIN}."
    echo "Assicurati che l'ambiente virtuale venv sia presente."
    exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents"
mkdir -p "${HOME}/.engramvault"

# Generate customized plist from template
sed \
    -e "s|__PYTHON_BIN__|${PYTHON_BIN}|g" \
    -e "s|__REMOTE_URL__|${REMOTE_URL}|g" \
    -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${PLIST_SRC}" > "${PLIST_DEST}"

# Unload previous instance if running
launchctl unload "${PLIST_DEST}" 2>/dev/null || true

# Load new instance
launchctl load "${PLIST_DEST}"

echo "✓ LaunchAgent installato e attivato con successo!"
echo "Log disponibili in:"
echo "  - ${HOME}/.engramvault/claude_code_sync.log"
echo "  - ${HOME}/.engramvault/claude_code_sync.err.log"
