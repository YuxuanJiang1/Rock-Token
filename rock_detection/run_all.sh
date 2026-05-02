#!/bin/bash
set -e

SAMPLES=(50 100 200 300 400 500)
STUDENTS=(onpolicy offpolicy)

for student in "${STUDENTS[@]}"; do
    for n in "${SAMPLES[@]}"; do
        echo "================================================"
        echo "Running: student=${student}  samples=${n}"
        echo "================================================"
        python rock_server.py --student "$student" --samples "$n"
    done
done

echo ""
echo "All runs complete."
