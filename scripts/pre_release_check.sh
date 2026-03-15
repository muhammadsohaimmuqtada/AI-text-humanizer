#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Compiling"
python3 -m compileall aip tests >/dev/null

echo "[2/6] Running tests"
python3 -m unittest discover -s tests -p 'test_*.py' -v >/dev/null

echo "[3/6] Checking release-critical files"
for f in README.md LICENSE SECURITY.md DEPLOYMENT.md CONTRIBUTING.md CODE_OF_CONDUCT.md CHANGELOG.md; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file: $f"
    exit 1
  fi
done

echo "[4/6] Scanning for machine-specific paths"
if grep -RIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='pre_release_check.sh' -E '/home/[A-Za-z0-9._-]+/|C:\\Users\\' .; then
  echo "ERROR: machine-specific path found."
  exit 1
fi

echo "[5/6] Scanning for high-risk secret patterns"
if grep -RIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='pre_release_check.sh' -E 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,}' .; then
  echo "ERROR: possible secret material found."
  exit 1
fi

# Reject accidental hardcoded real API key in config-like files.
if grep -RIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='pre_release_check.sh' -E '^AIP_API_KEY=' . | grep -v '.env.example' | grep -v 'AIP_API_KEY=change-me'; then
  echo "ERROR: non-placeholder AIP_API_KEY assignment found in repository files."
  exit 1
fi

echo "[6/6] Checking preflight command execution"
python3 -m aip.cli preflight --pretty >/dev/null

echo "Pre-release checks passed."
