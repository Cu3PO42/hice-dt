#!/bin/bash

find "$1" -type f -name '*.json' -mindepth 2 -printf '%P\n' | while read file; do
    if ! diff "$1/$file" "$2/$file"; then
        echo $file
    fi
done