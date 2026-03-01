#!/bin/bash
# Profile test durations to diagnose slow tests (Issue #1633)
#
# Usage: ./scripts/profile_tests.sh [options]
#
# Options:
#   -m MARKER    Pytest marker filter (default: "not production_server and not test_server")
#   -d DURATION  Number of slowest durations to show, 0 for all (default: 20)
#   -t TIMEOUT   Per-test timeout in seconds (default: 300)
#   -o OUTPUT    Output file path for the report (default: test_durations_report.txt)
#
# Examples:
#   ./scripts/profile_tests.sh
#   ./scripts/profile_tests.sh -m "production_server" -d 0 -t 600
#   ./scripts/profile_tests.sh -m "sklearn" -o sklearn_report.txt

set -euo pipefail

# Default values
MARKER_FILTER="not production_server and not test_server"
DURATIONS=20
TIMEOUT=300
OUTPUT_FILE="test_durations_report.txt"

# Parse command line arguments
while getopts "m:d:t:o:" opt; do
  case $opt in
    m) MARKER_FILTER="$OPTARG" ;;
    d) DURATIONS="$OPTARG" ;;
    t) TIMEOUT="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    *) echo "Usage: $0 [-m marker] [-d durations] [-t timeout] [-o output_file]" && exit 1 ;;
  esac
done

echo "=== OpenML Test Duration Profiler ==="
echo "Marker filter: $MARKER_FILTER"
echo "Durations to show: $DURATIONS"
echo "Timeout per test: ${TIMEOUT}s"
echo "Output file: $OUTPUT_FILE"
echo ""

pytest \
  --durations="$DURATIONS" \
  --timeout="$TIMEOUT" \
  -q \
  -m "$MARKER_FILTER" \
  2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "=== Report saved to $OUTPUT_FILE ==="
