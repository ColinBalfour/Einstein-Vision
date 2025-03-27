#!/bin/bash

# Exit on any error
set -e

# Define paths
CALIB_MAT="P3Data/Calib/calibration.mat"
CALIB_STRUCT_MAT="P3Data/Calib/calibration_struct.mat"
MATLAB_SCRIPT_DIR="Code"
PYTHON_SCRIPT="Code/data_extraction.py"

echo "=== [1/2] Converting calibration file in MATLAB... ==="
matlab -batch "addpath('$MATLAB_SCRIPT_DIR'); convert_calibration('$CALIB_MAT', '$CALIB_STRUCT_MAT')"

echo "=== [2/2] Running Python script to process calibration and images... ==="
python3 "$PYTHON_SCRIPT"

echo "✅ Done."