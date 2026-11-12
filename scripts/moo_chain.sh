#!/usr/bin/env bash
# Run a steering round's stages in order, outside any session, so an app restart cannot stop the chain:
#   bash scripts/moo_chain.sh steer12 train query baselines heldout report
# Each stage's start and end go to data/moo/<round>.log. A finished stage leaves data/moo/<round>.done.<stage> and is skipped on a
# rerun (MOO_STAGE_ARGS="--start 4" resumes a stage at a profile). A file data/moo/STOP ends the chain before the next stage (never in the middle of one).
# MOO_PRE_MAX=0.75 makes every stage wait for 10 minutes of readings under that commit fraction with no 50droid worker before
# it starts; MOO_LAST_START=1830 ends the chain (exit 3) instead of starting a stage after that local time.
# MOO_PRE_GPU_MB=12000 adds a third condition to that wait: at least that much GPU memory free, so a stage never starts into a
# GPU another job is using.
# On Windows a guard checks every 15 s and kills the running stage (logged, exit 2) if system commit passes MOO_MAX_COMMIT of
# its limit (default 0.92) or a 50droid worker is alive: the maintainer's own session on this machine always has priority.
cd "$(dirname "$0")/.." || exit 1
round=$1; shift
log=data/moo/$round.log
py=.venv/Scripts/python; [ -x "$py" ] || [ -x "$py.exe" ] || py=.venv/bin/python
export BONSAI_GPU_FRACTION=${BONSAI_GPU_FRACTION:-0.75} PYTHONUNBUFFERED=1
mkdir -p data/moo
guard() { command -v powershell >/dev/null 2>&1 && powershell -NoProfile -ExecutionPolicy Bypass -File scripts/moo_guard.ps1 -Max "${MOO_MAX_COMMIT:-0.92}" -MinFreeGpuMb "${MOO_GPU_MB:-0}" 2>/dev/null < /dev/null | tr -d '
'; }
if [ -z "${MOO_PRE_MAX:-}" ]; then g=$(guard); case "$g" in ok*|"") ;; *) echo "== GUARD: not starting $round: $g $(date '+%m-%d %H:%M:%S')" >> data/moo/$round.log; exit 2;; esac; fi  # with MOO_PRE_MAX each stage waits instead
echo "== chain $round: $* (pid $$) $(date '+%m-%d %H:%M:%S')" >> "$log"
for s in "$@"; do
  if [ -e "data/moo/$round.done.$s" ]; then echo "== $s already done" >> "$log"; continue; fi
  if [ -e data/moo/STOP ]; then echo "== STOP file: chain ends before $s $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 0; fi
  if [ -n "${MOO_PRE_MAX:-}" ]; then  # wait for 21 readings 30 s apart (10 minutes) that all pass the guard at MOO_PRE_MAX
    echo "== $s waits for 10 min of commit under $MOO_PRE_MAX and no 50droid worker $(date '+%m-%d %H:%M:%S')" >> "$log"
    n=0
    while [ $n -lt 21 ]; do
      [ -e data/moo/STOP ] && { echo "== STOP file: chain ends before $s $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 0; }
      if [ -n "${MOO_LAST_START:-}" ] && [ "$(date +%H%M)" -gt "$MOO_LAST_START" ]; then
        echo "== too late to start $s (after $MOO_LAST_START) $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 3; fi
      g=$(MOO_MAX_COMMIT=$MOO_PRE_MAX MOO_GPU_MB=${MOO_PRE_GPU_MB:-0} guard)
      case "$g" in ok*) n=$((n+1));; *) [ $n -gt 0 ] && echo "   wait reset: $g $(date '+%H:%M:%S')" >> "$log"; n=0;; esac
      [ $n -lt 21 ] && sleep 30
    done
  fi
  echo "== $s start $(date '+%m-%d %H:%M:%S')" >> "$log"
  "$py" -m "bonsai_interp.moo.$round" "$s" ${MOO_STAGE_ARGS:-} >> "$log" 2>&1 &
  job=$!
  while kill -0 $job 2>/dev/null; do
    sleep 15
    g=$(guard)
    case "$g" in ok*|"") ;; *)
      kill -0 $job 2>/dev/null || break
      echo "== GUARD killed $s: $g $(date '+%m-%d %H:%M:%S')" >> "$log"
      taskkill //PID "$(cat /proc/$job/winpid)" //T //F >> "$log" 2>&1; wait $job; exit 2;;
    esac
  done
  if wait $job; then
    echo "== $s end $(date '+%m-%d %H:%M:%S')" >> "$log"; touch "data/moo/$round.done.$s"
  else
    echo "== FAILED $s $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 1
  fi
done
echo "== ALL DONE $(date '+%m-%d %H:%M:%S')" >> "$log"
