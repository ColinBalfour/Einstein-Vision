#!/bin/bash

# SBATCH --mail-type=ALL
# SBATCH -n 10
# SBATCH --mem=20000
# SBATCH --gres=gpu:1
# SBATCH --constraint=V100|A100|L40S
# SBATCH -p short
# SBATCH -t 23:59:00
# SBATCH --mail-user=cbalfour@wpi.edu

source ~/.bashrc

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
  python3 -u Code/ImageModels/MetricDepth.py "$i"
done

