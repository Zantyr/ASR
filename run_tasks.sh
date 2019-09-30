#!/bin/bash

CACHE_PATH="/pictec/models"
CMD="$1"
VIOLENT="True"  # need to be a Python constant

if [ "$CMD" = "stats" ]; then
echo "Running the statistics summary!"
python3 -c '
import os, sys, syntax
sys.path.append("tasks")
import core
for i in os.listdir("tasks"):
    if i.endswith(".py") and i != "core.py":
        __import__(i.split(".")[0])
cache = "'"$CACHE_PATH"'"
state = syntax.utils.PersistentState(cache + "/state")
for task in core.Task.all():
    if task.name not in state.keys():
        state[task.name] = task.name
    this_cache = os.path.join(cache, state[task.name])
    print(task.summary(this_cache))
';
else
echo "Running the full task run - continuing from the last stop point"
python3 -c '
import os, sys, syntax
sys.path.append("tasks")
violent = '$VIOLENT'
import core
for i in os.listdir("tasks"):
    if i.endswith(".py") and i != "core.py":
        __import__(i.split(".")[0])
cache = "'"$CACHE_PATH"'"
state = syntax.utils.PersistentState(cache + "/state")
for task in core.Task.all():
    try:
        if task.name not in state.keys():
            state[task.name] = task.name
        this_cache = os.path.join(cache, state[task.name])
        if not task.validate(this_cache):
            task.run(this_cache)
    except Exception as e:
        print("Task {} failed".format(task.name))
        if violent:
            raise e

# Fetch all results...
for task in core.Task.all():
    if task.name not in state.keys():
        state[task.name] = task.name
    this_cache = os.path.join(cache, state[task.name])
    print(task.summary(this_cache))
';
fi