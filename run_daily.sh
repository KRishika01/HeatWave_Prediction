#!/bin/bash

# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Automated Daily Run Script
# ========================================================================

# Get the directory where the script is located
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

# Log file path
LOG_FILE="$PROJECT_DIR/data/cron_logs.log"
mkdir -p "$PROJECT_DIR/data"

echo "------------------------------------------------------------" >> "$LOG_FILE"
echo "RUN STARTED: $(date)" >> "$LOG_FILE"
echo "------------------------------------------------------------" >> "$LOG_FILE"

# Run the prediction pipeline for all cities
# Using python3 specifically to ensure the right version is used
/usr/bin/python3 step7_realtime_api.py --city all >> "$LOG_FILE" 2>&1

echo "------------------------------------------------------------" >> "$LOG_FILE"
echo "RUN FINISHED: $(date)" >> "$LOG_FILE"
echo "------------------------------------------------------------" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
