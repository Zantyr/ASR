#!/bin/bash

set -u

failed=0
passed=0

python3 -c '
from fwk.stage_meta import ToDo
import fwk.stage
ToDo.status()'

for file in `ls tests`; do
	printf "$file"
	python3 tests/$file &> /dev/null
	if [[ "$?" -ne "0" ]]; then
		failed=$(expr $failed + 1)
		printf " - FAILED!\n"
	else
		passed=$(expr $passed + 1)
		printf " - PASSED!\n"
	fi
done

echo "All tests run, passed: $passed, failed: $failed"
