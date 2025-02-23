#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR=$1

if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

FILES=($(ls -v $DIR/*.npy))

if [ ${#FILES[@]} -lt 2 ]; then
    echo "Not enough .npy files in the directory to compare."
    exit 1
fi

for ((i = 0; i < ${#FILES[@]} - 1; i++)); do
    FILE1=${FILES[$i]}
    FILE2=${FILES[$i + 1]}
    python3 diff_npy.py "$FILE1" "$FILE2"
done
