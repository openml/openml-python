#!/bin/bash
# Profile test durations to diagnose slow tests (Issue #1633)
# Usage: ./scripts/profile_tests.sh [marker_filter]
#
# Examples:
#   ./scripts/profile_tests.sh                               # non-server tests
#   ./scripts/profile_tests.sh "production_server"            # production server tests only
#   ./scripts/profile_tests.sh "sklearn"                      # sklearn tests only

set -euo pipefail

MARKER_FILTER="${1:-not production_server and not test_server}"

echo "=== OpenML Test Duration Profiler ==="
echo "Marker filter: $MARKER_FILTER"
echo "Timeout per test: 300s"
echo ""

pytest \
  --durations=0 \
  --timeout=300 \
  -q \
  -m "$MARKER_FILTER" \
  2>&1 | tee test_durations_report.txt

echo ""
echo "=== Report saved to test_durations_report.txt ==="
