#!/usr/bin/env bash
# Run a steering round's stages in order, outside any session, so an app restart cannot stop the chain:
#   bash scripts/moo_chain.sh steer12 train query baselines heldout report
# Each stage's start and end go to data/moo/<round>.log. A finished stage leaves data/moo/<round>.done.<stage> and is skipped on a
# rerun. A file data/moo/STOP ends the chain before the next stage (never in the middle of one).
cd "$(dirname "$0")/.." || exit 1
round=$1; shift
log=data/moo/$round.log
py=.venv/Scripts/python; [ -x "$py" ] || [ -x "$py.exe" ] || py=.venv/bin/python
export BONSAI_GPU_FRACTION=${BONSAI_GPU_FRACTION:-0.75} PYTHONUNBUFFERED=1
mkdir -p data/moo
echo "== chain $round: $* (pid $$) $(date '+%m-%d %H:%M:%S')" >> "$log"
for s in "$@"; do
  if [ -e "data/moo/$round.done.$s" ]; then echo "== $s already done" >> "$log"; continue; fi
  if [ -e data/moo/STOP ]; then echo "== STOP file: chain ends before $s $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 0; fi
  echo "== $s start $(date '+%m-%d %H:%M:%S')" >> "$log"
  if "$py" -m "bonsai_interp.moo.$round" "$s" >> "$log" 2>&1; then
    echo "== $s end $(date '+%m-%d %H:%M:%S')" >> "$log"; touch "data/moo/$round.done.$s"
  else
    echo "== FAILED $s $(date '+%m-%d %H:%M:%S')" >> "$log"; exit 1
  fi
done
echo "== ALL DONE $(date '+%m-%d %H:%M:%S')" >> "$log"
