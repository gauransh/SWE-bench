#!/usr/bin/env bash

# Configuration
DATASET_NAME="/data1/gtandon/SWE-bench/data/swe-bench.json"
PREDICTIONS_PATH="/data1/gtandon/acr_new_instance/results/acr-run-1/predictions_for_swebench.json"
RUN_ID="swe-bench-lite"
MAX_WORKERS=1
FORCE_REBUILD="False"
CACHE_LEVEL="env"
CLEAN="False"
TIMEOUT=1800

LOG_FILE="evaluation_run_$(date +%Y-%m-%d_%H-%M-%S).log"
PID_FILE="evaluation_run.pid"

start() {
    # Check if process is already running
    if [ -f "$PID_FILE" ]; then
        if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            echo "Process is already running with PID $(cat "$PID_FILE")."
            exit 1
        else
            # Stale PID file; remove it
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting evaluation in the background..."
    nohup python3 -m swebench.harness.run_evaluation \
        --dataset_name "$DATASET_NAME" \
        --predictions_path "$PREDICTIONS_PATH" \
        --run_id "$RUN_ID" \
        --max_workers "$MAX_WORKERS" \
        --force_rebuild "$FORCE_REBUILD" \
        --cache_level "$CACHE_LEVEL" \
        --clean "$CLEAN" \
        --timeout "$TIMEOUT" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Process started with PID: $(cat "$PID_FILE")"
    echo "Logs are being written to: $LOG_FILE"
}

stop() {
    # Check if PID file exists
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Is the process running?"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping process with PID $PID..."
        kill $PID
        rm -f "$PID_FILE"
        echo "Process stopped."
    else
        echo "No process found with PID $PID. Removing stale PID file."
        rm -f "$PID_FILE"
    fi
}

# Check command line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 {start|stop}"
    exit 1
fi

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "Invalid command: $1"
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac

