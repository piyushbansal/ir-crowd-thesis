#!/bin/bash
# Dumb way to kill all jobs on LSF
while :
do
  bkill -u pbansal
  sleep 4s
done
