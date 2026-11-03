# Source in a training sbatch, then: start_watchdog <log file>
# When a vLLM worker or Ray task dies, verl often hangs instead of exiting (Nov 3 smoke runs crashed
# ~15 min in and then sat idle until the time limit). This cancels the job once a fatal error shows up
# in the log, so a crash costs minutes of GPU time instead of hours.
start_watchdog() {
  local log="$1"
  (
    while sleep 60; do
      if grep -qE "WorkerProc hit an exception|Error executing job with overrides|RayTaskError|OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null; then
        echo "[watchdog] fatal error found in $log; cancelling job ${SLURM_JOB_ID:-?} in 60 s" | tee -a "$log"
        sleep 60   # let the traceback finish printing
        scancel "${SLURM_JOB_ID}"
        exit 0
      fi
    done
  ) &
}
