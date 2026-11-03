#!/usr/bin/env bash
cd "$HOME/githubs/mimir-weight-space"
.venv/bin/python src/accept.py --dataset scala-en --device cuda:0 --out results/accept_base
.venv/bin/python src/accept.py --dataset dala --device cuda:0 --out results/accept_base
