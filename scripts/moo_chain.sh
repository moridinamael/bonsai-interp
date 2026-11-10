#!/usr/bin/env bash
# Run a steering round's stages in order, outside any session, so an app restart cannot stop the chain:
#   bash scripts/moo_chain.sh steer12 train query baselines heldout report
# Each stage's start and end go to data/moo/<round>.log. A finished stage leaves data/moo/<round>.done.<stage> and is skipped on a
# rerun. A file data/moo/STOP ends the chain before the next stage (never in the middle of one).
# On Windows a guard checks every 15 s and kills the running stage (logged, exit 2) if system commit passes MOO_MAX_COMMIT of
# its limit (default 0.92) or a 50droid worker is alive: the maintainer's own session on this machine always has priority.
cd "$(dirname "$0")/.." || exit 1
round=$1; shift
log=data/moo/$round.log
py=.venv/Scripts/python; [ -x "$py" ] || [ -x "$py.exe" ] || py=.venv/bin/python
export BONSAI_GPU_FRACTION=${BONSAI_GPU_FRACTION:-0.75} PYTHONUNBUFFERED=1
mkdir -p data/moo
guard() { command -v powershell >/dev/null 2>&1 && powershell -NoProfile -ExecutionPolicy Bypass -File scripts/moo_guard.ps1 -Max "${MOO_MAX_COMMIT:-0.92}" 2>/dev/null < /dev/null | tr -d ''; }
g=$(guard); case "$g" in ok*|"") ;; *) echo "== GUARD: not starting $round: $g $(date '+%m-%d %H:%M:%S')" >> data/moo/$round.log; exit 2;; esac
echo "== chain $round: $* (pid $$) $(date '+%m-%d %H:%M:%S')" >> "$log"
for s in "$@"; do
  if [ -e "data/moo/$round.done.$s" ]; then echo "== $s already done" >> "$log"; continue; fi
  if [ -e data/moo/STOP ]; then echo "== STOP file: chain ends before $s $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 0; fi
  echo "== $s start $(date '+%m-%d %H:%M:%S')" >> "$log"
  "$py" -m "bonsai_interp.moo.$round" "$s" >> "$log" 2>&1 &
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
