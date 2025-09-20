#!/bin/bash

# A script to run CTC segmentation, verify segments, and cut audio.

# --- Configuration and Argument Parsing ---
set -e # Exit immediately if a command exits with a non-zero status.

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root_dir> <config_yaml>"
    exit 1
fi

ROOT_DIR="$1"
CONFIG_FILE="$2"

# --- Prerequisite Check ---
# Check if yq is installed, as it's required for parsing the YAML config.
if ! command -v yq &> /dev/null; then
    echo "Error: 'yq' is not installed. Please install it to continue."
    echo "Installation instructions: https://github.com/mikefarah/yq/"
    exit 1
fi

# --- Path and Log Setup ---
OUTPUT_DIR="${ROOT_DIR}/output"
LOG_DIR="${OUTPUT_DIR}/logs"
PROCESSED_AUDIO_DIR="${OUTPUT_DIR}/processed/audio"
PROCESSED_TEXT_DIR="${OUTPUT_DIR}/processed/text"

# Create necessary directories
mkdir -p "${LOG_DIR}"

# --- Read Configuration from YAML ---
get_config() {
    local value
    value=$(yq -r "$1" "$CONFIG_FILE")
    if [ "$value" == "null" ]; then
        echo "Error: Key '$1' not found in config file '$CONFIG_FILE'." >&2
        exit 1
    fi
    echo "$value"
}

# General configs
SAMPLE_RATE=$(get_config '.sample_rate')
NEMO_MODEL=$(get_config '.models.nemo_model')
THRESHOLD=$(get_config '.threshold')
OFFSET=$(get_config '.offset')
OUTPUT_FORMAT=$(get_config '.output_format')
MAX_DURATION=$(get_config '.max_duration')

# --- Step 4: CTC Segmentation Pipeline ---
echo "----- [Step 4] Running CTC Segmentation Pipeline -----"

# Run CTC Segmentation for different window lengths
for window in 8000 12000; do
    echo "----- Running CTC Segmentation with window size ${window} -----"
    LOG_CTC_SEGMENTATION="${LOG_DIR}/ctc_segmentation_${window}.log"
    python -m scripts.ctc_segmentation.run_ctc_segmentation \
        --output_dir="${OUTPUT_DIR}" \
        --audio="${PROCESSED_AUDIO_DIR}" \
        --text="${PROCESSED_TEXT_DIR}" \
        --sample_rate="${SAMPLE_RATE}" \
        --model="${NEMO_MODEL}" \
        --window_len=${window} > "${LOG_CTC_SEGMENTATION}" 2>&1
    echo "CTC Segmentation with window size ${window} complete. Log saved to ${LOG_CTC_SEGMENTATION}"
done

# Verify Segments
echo "----- Verifying Segments -----"
LOG_VERIFY_SEGMENTS="${LOG_DIR}/verify_segments.log"
python -m scripts.ctc_segmentation.verify_segments \
    --base_dir="${OUTPUT_DIR}" > "${LOG_VERIFY_SEGMENTS}" 2>&1
echo "Segment verification complete. Log saved to ${LOG_VERIFY_SEGMENTS}"

# Cut Audio
echo "----- Cutting Audio -----"
LOG_CUT_AUDIO="${LOG_DIR}/cut_audio.log"
python -m scripts.ctc_segmentation.cut_audio \
    --output_dir="${OUTPUT_DIR}" \
    --alignment="${OUTPUT_DIR}/verified_segments" \
    --threshold="${THRESHOLD}" \
    --offset="${OFFSET}" \
    --output_format="${OUTPUT_FORMAT}" \
    --sample_rate="${SAMPLE_RATE}" \
    --max_duration="${MAX_DURATION}" > "${LOG_CUT_AUDIO}" 2>&1
echo "Audio cutting complete. Log saved to ${LOG_CUT_AUDIO}"

echo "CTC Segmentation pipeline (Step 4) is complete. ✅"
