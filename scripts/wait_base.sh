#!/usr/bin/env bash
# Block until the dev baseline finishes, then print the summaries.
cd "$HOME/githubs/mimir-weight-space"
while systemctl --user is-active --quiet ws-base-dev-dan || systemctl --user is-active --quiet ws-base-dev-eng; do
  sleep 20
done
grep -h "n=" logs/base_dev_dan.log logs/base_dev_eng.log
