#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <start_scene> <end_scene>"
  exit 1
fi

start=$1
end=$2

for (( i=start; i<=end; i++ ))
do
  echo "Running scene $i..."
  mkdir -p outputs/depth/scene_$i
  python3 Code/ImageModels/MetricDepth.py "$i"
done

