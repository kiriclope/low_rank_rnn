#!/bin/bash
# Publish a sweep's figures to Leon's gallery under a MEANINGFUL title.
# Title comes from results/dual/<sweep>/TITLE (falls back to the raw dir name).
# Classification uses the SOURCE SUBDIR, not the filename — flow panels are named traj_*.png.
set -u
sw="$1"; root=/home/leon/rnn/results/figures
title=$(cat "/home/leon/rnn/results/dual/$sw/TITLE" 2>/dev/null || echo "$sw")
cd "$root" || exit 1; [ -d "$sw" ] || { echo "no figures for $sw"; exit 1; }
rm -rf "$HOME/dual/rnn/$title"; n=0
while IFS= read -r f; do
  rel="${f#$sw/}"; name=$(basename "$f")
  case "$rel" in individual/*) seed=$(echo "$rel"|cut -d/ -f2); sub=$(echo "$rel"|cut -d/ -f3);; *) seed=summary; sub=$(echo "$rel"|cut -d/ -f2);; esac
  case "$sub" in flow|scatter|accuracy|traj) type=$sub ;;
    *.png) case "$name" in *accuracy*) type=accuracy;; *fp_*) type=flow;; *traj*) type=traj;; *) type=misc;; esac ;;
    *) type=$sub ;; esac
  dest="$HOME/dual/rnn/$title/$type"; mkdir -p "$dest"; cp "$f" "$dest/${seed}_${name}"; n=$((n+1))
done < <(find "$sw" -name '*.png')
echo "published $n PNGs -> ~/dual/rnn/$title"
