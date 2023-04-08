#!/usr/bin/env bash

unset LOCAL_DEBUG; python make_plots.py \
  --dataset "elasticc" \
  --architecture "tinho" \
  --model $1
  # --redshift ""
