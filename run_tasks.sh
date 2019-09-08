#!/bin/bash

CACHE_PATH="/pictec/models"

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
    if not task.validate(this_cache):
        task.run(this_cache)

# Fetch all results...
for task in core.Task.all():
    if task.name not in state.keys():
        state[task.name] = task.name
    this_cache = os.path.join(cache, state[task.name])
    print(task.summary(this_cache))
'
