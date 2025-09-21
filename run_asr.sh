#!/bin/bash

# A script to run ASR transcription using different models.

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
MANIFESTS_DIR="${OUTPUT_DIR}/manifests"
DATASET_MANIFEST="${MANIFESTS_DIR}/manifest.json"

# Create necessary directories
mkdir -p "${LOG_DIR}"
mkdir -p "${MANIFESTS_DIR}"

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
BATCH_SIZE=$(get_config '.batch_size')
EDGE_LEN=$(get_config '.edge_len')
DEVICE=$(get_config '.device')
NUM_WORKERS=$(get_config '.num_workers')

# Model paths
NEMO_MODEL=$(get_config '.models.nemo_model')
WAV2VEC2_MODEL=$(get_config '.models.wav2vec2_v3')
FASTER_WHISPER_MODEL=$(get_config '.models.faster_whisper_large')

# ASR module flags
RUN_NEMO=$(get_config '.asr_modules.nemo')
RUN_WAV2VEC2=$(get_config '.asr_modules.wav2vec2')
RUN_WHISPER=$(get_config '.asr_modules.whisper')
RUN_FASTER_WHISPER=$(get_config '.asr_modules.faster_whisper')


# --- Step 5: Transcribe Speech using Nemo ---
if [ "$RUN_NEMO" = "true" ]; then
    echo "----- [Step 5] Transcribing speech using Nemo -----"
    LOG_TRANSCRIBE_NEMO="${LOG_DIR}/transcribe_nemo.log"
    OUTPUT_MANIFEST_NEMO="${DATASET_MANIFEST/.json/_nemo.json}"

    python -m scripts.asr.transcribe_speech_nemo \
        --model_path="${NEMO_MODEL}" \
        --dataset_manifest="${DATASET_MANIFEST}" \
        --dataset_manifest_transcribed="${OUTPUT_MANIFEST_NEMO}" \
        --device="${DEVICE}" \
        --edge_len="${EDGE_LEN}" \
        --batch_size="${BATCH_SIZE}" > "${LOG_TRANSCRIBE_NEMO}" 2>&1

    echo "Nemo transcription complete. Log saved to ${LOG_TRANSCRIBE_NEMO}"
    echo ""
else
    echo "----- [Step 5] Skipping Nemo transcription -----"
    echo ""
fi


# --- Step 6: Transcribe Speech using Wav2Vec2 ---
if [ "$RUN_WAV2VEC2" = "true" ]; then
    echo "----- [Step 6] Transcribing speech using Wav2Vec2 -----"
    LOG_TRANSCRIBE_WAV2VEC2="${LOG_DIR}/transcribe_wav2vec2_v3.log"
    OUTPUT_MANIFEST_WAV2VEC2="${DATASET_MANIFEST/.json/_wav2vec2_v3.json}"

    python -m scripts.asr.transcribe_speech_wav2vec2 \
        --model_path="${WAV2VEC2_MODEL}" \
        --device="${DEVICE}" \
        --edge_len="${EDGE_LEN}" \
        --dataset_manifest="${DATASET_MANIFEST}" \
        --dataset_manifest_transcribed="${OUTPUT_MANIFEST_WAV2VEC2}" > "${LOG_TRANSCRIBE_WAV2VEC2}" 2>&1

    echo "Wav2Vec2 transcription complete. Log saved to ${LOG_TRANSCRIBE_WAV2VEC2}"
    echo ""
else
    echo "----- [Step 6] Skipping Wav2Vec2 transcription -----"
    echo ""
fi


# --- Step 7: Transcribe Speech using Faster Whisper ---
if [ "$RUN_FASTER_WHISPER" = "true" ]; then
    echo "----- [Step 7] Transcribing speech using Faster Whisper -----"
    LOG_TRANSCRIBE_FASTER_WHISPER="${LOG_DIR}/transcribe_faster_whisper_large.log"
    OUTPUT_MANIFEST_FASTER_WHISPER="${DATASET_MANIFEST/.json/_faster_whisper_large.json}"

    python -m scripts.asr.transcribe_speech_faster_whisper \
        --model_path="${FASTER_WHISPER_MODEL}" \
        --batch_size="${BATCH_SIZE}" \
        --edge_len="${EDGE_LEN}" \
        --device="${DEVICE}" \
        --dataset_manifest="${DATASET_MANIFEST}" \
        --dataset_manifest_transcribed="${OUTPUT_MANIFEST_FASTER_WHISPER}" > "${LOG_TRANSCRIBE_FASTER_WHISPER}" 2>&1

    echo "Faster Whisper transcription complete. Log saved to ${LOG_TRANSCRIBE_FASTER_WHISPER}"
    echo ""
else
    echo "----- [Step 7] Skipping Faster Whisper transcription -----"
    echo ""
fi

# --- Step 8: Transcribe Speech using Whisper ---
if [ "$RUN_WHISPER" = "true" ]; then
    echo "----- [Step 8] Transcribing speech using Whisper -----"
    LOG_TRANSCRIBE_WHISPER="${LOG_DIR}/transcribe_whisper_large.log"
    OUTPUT_MANIFEST_WHISPER="${DATASET_MANIFEST/.json/_whisper_large.json}"
    WHISPER_MODEL=$(get_config '.models.whisper_large')

    python -m scripts.asr.transcribe_speech_whisper \
        --model_path="${WHISPER_MODEL}" \
        --batch_size="${BATCH_SIZE}" \
        --edge_len="${EDGE_LEN}" \
        --device="${DEVICE}" \
        --dataset_manifest="${DATASET_MANIFEST}" \
        --dataset_manifest_transcribed="${OUTPUT_MANIFEST_WHISPER}" > "${LOG_TRANSCRIBE_WHISPER}" 2>&1

    echo "Whisper transcription complete. Log saved to ${LOG_TRANSCRIBE_WHISPER}"
    echo ""
else
    echo "----- [Step 8] Skipping Whisper transcription -----"
    echo ""
fi

# --- Completion ---
echo "ASR transcription steps are complete. ✅"
