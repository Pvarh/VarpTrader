#!/bin/bash
# Deploy code changes to VPS. NEVER overwrites config.json or runtime data.
# Usage: ./deploy.sh [file1 file2 ...]
#   No args: syncs all .py files + templates
#   With args: syncs only the specified files

set -euo pipefail

VPS="root@162.55.50.28"
REMOTE_DIR="~/Varptrader/autotrader"
LOCAL_DIR="$(cd "$(dirname "$0")/autotrader" && pwd)"

# Files that must NEVER be overwritten on VPS (VPS is source of truth)
NEVER_SYNC=(
    "config.json"
    "data/"
    "logs/"
    "overseer/reports/"
    "overseer/change_log.json"
    "overseer/strategy_log.json"
    "weekly_bias.json"
    ".env"
)

is_blocked() {
    local file="$1"
    for blocked in "${NEVER_SYNC[@]}"; do
        if [[ "$file" == "$blocked" || "$file" == "$blocked"* ]]; then
            echo "BLOCKED: $file (VPS is source of truth)"
            return 0
        fi
    done
    return 1
}

if [ $# -gt 0 ]; then
    # Deploy specific files
    for file in "$@"; do
        # Strip leading autotrader/ if present
        file="${file#autotrader/}"
        if is_blocked "$file"; then
            continue
        fi
        echo "Deploying: $file"
        scp "$LOCAL_DIR/$file" "$VPS:$REMOTE_DIR/$file"
    done
else
    # Deploy all Python files + templates (excluding blocked files)
    echo "Syncing all code to VPS..."
    rsync -avz --progress \
        --chown=root:root \
        --include='*.py' \
        --include='*.html' \
        --include='*/' \
        --exclude='*' \
        --exclude='config.json' \
        --exclude='data/' \
        --exclude='logs/' \
        --exclude='.env' \
        --exclude='overseer/reports/' \
        --exclude='overseer/change_log.json' \
        --exclude='overseer/strategy_log.json' \
        --exclude='weekly_bias.json' \
        "$LOCAL_DIR/" "$VPS:$REMOTE_DIR/"

    # Restore config.json ACL so trader user (overseer) can write it
    ssh "$VPS" "setfacl -m u:trader:rw,m::rw $REMOTE_DIR/config.json 2>/dev/null || true"
fi

echo ""
echo "To pull VPS config locally: scp $VPS:$REMOTE_DIR/config.json $LOCAL_DIR/config.json"
echo "To run tests: ssh $VPS \"cd $REMOTE_DIR && docker compose exec autotrader python -m pytest tests/ -q\""
echo "To rebuild:   ssh $VPS \"cd $REMOTE_DIR && docker compose build autotrader && docker compose up -d autotrader\""
