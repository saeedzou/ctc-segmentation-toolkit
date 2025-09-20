#!/bin/bash

# A script to preprocess audio and text data based on a YAML config.
# Replicates modes 0 and 1 of the main Python pipeline script.

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
AUDIO_DIR="${ROOT_DIR}/audio"
TEXT_DIR="${ROOT_DIR}/text"
OUTPUT_DIR="${ROOT_DIR}/output"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create necessary directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}/processed/audio"
mkdir -p "${OUTPUT_DIR}/processed/text"

# Log file paths
LOG_PREPARE_AUDIO="${LOG_DIR}/prepare_audio.log"
LOG_PREPARE_TEXTS="${LOG_DIR}/prepare_texts.log"

# --- Read Configuration from YAML ---
# This function safely reads a value from the YAML file.
# yq will return the string 'null' for missing keys, which we handle.
get_config() {
    local value
    value=$(yq -r "$1" "$CONFIG_FILE") # <-- The fix is here
    if [ "$value" == "null" ]; then
        echo "Error: Key '$1' not found in config file '$CONFIG_FILE'." >&2
        exit 1
    fi
    echo "$value"
}

# General configs
SAMPLE_RATE=$(get_config '.sample_rate')
LANG_ID=$(get_config '.lang_id')
CUT_PREFIX=$(get_config '.cut_prefix')
ADDITIONAL_SPLIT_SYMBOLS=$(get_config '.additional_split_symbols')

# Text processing boolean flags
REMOVE_BRACKETS=$(get_config '.remove_brackets')
REMOVE_ASTERISKS=$(get_config '.remove_asterisks')
REMOVE_PARENTHESES=$(get_config '.remove_parentheses')
REMOVE_NUM_IN_BRACKETS=$(get_config '.remove_number_in_brackets')
REMOVE_NUM_IN_CURLY=$(get_config '.remove_number_in_curly_brackets')
REMOVE_NUM_IN_PARENS=$(get_config '.remove_number_in_parentheses')
REMOVE_NUM_IN_GUILL=$(get_config '.remove_number_in_guillemets')
REMOVE_SPEAKER_LABELS=$(get_config '.remove_speaker_labels')
SPLIT_USING_PATTERN=$(get_config '.split_using_pattern')
SPLIT_USING_SAT=$(get_config '.split_using_sat')
SPLIT_ON_QUOTES=$(get_config '.split_on_quotes')
SPLIT_ON_VERBS=$(get_config '.split_on_verbs')
SPLIT_ON_VERBS_MIN_WORDS=$(get_config '.split_on_verbs_min_words')
SPLIT_ON_VERBS_MAX_WORDS=$(get_config '.split_on_verbs_max_words')


# Model paths
POS_TAGGER_PATH=$(get_config '.models.pos_tagger')
NEMO_MODEL=$(get_config '.models.nemo_model')

# --- Step 0: Prepare Audio Files ---
echo "----- [Step 0] Preparing audio files in ${AUDIO_DIR} -----"

python -m scripts.preprocessing.prepare_data \
    --audio_dir="${AUDIO_DIR}" \
    --output_dir="${OUTPUT_DIR}/processed/audio" \
    --sample_rate="${SAMPLE_RATE}" > "${LOG_PREPARE_AUDIO}" 2>&1

echo "Audio preparation complete. Log saved to ${LOG_PREPARE_AUDIO}"
echo ""

# --- Step 1: Prepare Text Files ---
echo "----- [Step 1] Preparing text files in ${TEXT_DIR} -----"

python -m scripts.preprocessing.prepare_data \
    --in_text="${TEXT_DIR}" \
    --output_dir="${OUTPUT_DIR}/processed/text" \
    --remove_brackets="${REMOVE_BRACKETS}" \
    --remove_asterisks="${REMOVE_ASTERISKS}" \
    --remove_parentheses="${REMOVE_PARENTHESES}" \
    --remove_number_in_brackets="${REMOVE_NUM_IN_BRACKETS}" \
    --remove_number_in_curly_brackets="${REMOVE_NUM_IN_CURLY}" \
    --remove_number_in_parentheses="${REMOVE_NUM_IN_PARENS}" \
    --remove_number_in_guillumets="${REMOVE_NUM_IN_GUILL}" \
    --remove_speaker_labels="${REMOVE_SPEAKER_LABELS}" \
    --split_using_pattern="${SPLIT_USING_PATTERN}" \
    --split_using_sat="${SPLIT_USING_SAT}" \
    --split_on_quotes="${SPLIT_ON_QUOTES}" \
    --split_on_verbs="${SPLIT_ON_VERBS}" \
    --pos_tagger_path="${POS_TAGGER_PATH}" \
    --split_on_verbs_min_words="${SPLIT_ON_VERBS_MIN_WORDS}" \
    --split_on_verbs_max_words="${SPLIT_ON_VERBS_MAX_WORDS}" \
    --language="${LANG_ID}" \
    --cut_prefix="${CUT_PREFIX}" \
    --model="${NEMO_MODEL}" \
    --sample_rate="${SAMPLE_RATE}" \
    --additional_split_symbols="${ADDITIONAL_SPLIT_SYMBOLS}" > "${LOG_PREPARE_TEXTS}" 2>&1

echo "Text preparation complete. Log saved to ${LOG_PREPARE_TEXTS}"
echo ""

# --- Step 2: Transcribe Long Speech ---
echo "----- [Step 2] Transcribing long audio files -----"

PROCESSED_AUDIO_DIR="${OUTPUT_DIR}/processed/audio"
TRANSCRIPT_DIR="${OUTPUT_DIR}/processed/hypothesis"
LOG_TRANSCRIBE_LONG="${LOG_DIR}/transcribe_long_speech_nemo.log"

mkdir -p "${TRANSCRIPT_DIR}"

# Read additional config values for this step
BATCH_SIZE=$(get_config '.batch_size')
DEVICE=$(get_config '.device')

python -m scripts.asr.transcribe_long_speech_nemo \
    --model_path="${NEMO_MODEL}" \
    --batch_size="${BATCH_SIZE}" \
    --device="${DEVICE}" \
    --audio_dir="${PROCESSED_AUDIO_DIR}" \
    --transcript_dir="${TRANSCRIPT_DIR}" > "${LOG_TRANSCRIBE_LONG}" 2>&1

echo "Long audio transcription complete. Log saved to ${LOG_TRANSCRIBE_LONG}"
echo ""

# --- Step 3: Remove Outliers ---
echo "----- [Step 3] Removing outliers -----"

LONG_WER_THRESHOLD=$(get_config '.long_wer_threshold')
LONG_CER_THRESHOLD=$(get_config '.long_cer_threshold')

python -m scripts.preprocessing.remove_outliers \
    --output_dir="${OUTPUT_DIR}" \
    --min_wer="${LONG_WER_THRESHOLD}" \
    --min_cer="${LONG_CER_THRESHOLD}"

echo "Outlier removal complete."
echo ""

# --- Completion ---
echo "Preprocessing steps 0, 1, 2, and 3 are complete. ✅"
