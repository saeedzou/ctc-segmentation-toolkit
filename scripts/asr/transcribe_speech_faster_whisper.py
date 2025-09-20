import argparse
import json
import torch
import os
from faster_whisper import WhisperModel
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.utils import logging
from tqdm import tqdm
from scripts.preprocessing.normalizer.calculate_metrics import calculate_metrics


def clean_manifest(file_path):
    valid_lines = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON at line {idx} in {file_path}")

    with open(file_path, 'w') as f:
        f.writelines(valid_lines)

# Load faster_whisper model
def load_model(model_path, device, device_index, compute_type):
    """Loads the faster_whisper model."""
    logging.info(f"Loading faster_whisper model: {model_path} with compute_type: {compute_type} on device: {device}:{device_index}")
    model = WhisperModel(model_path, device=device, device_index=device_index, compute_type=compute_type)
    return model

# faster_whisper handles audio loading and processing internally, so process_audio_files_batch is not needed.


def transcribe_audio(model, dataset_manifest, dataset_manifest_transcribed, edge_len, batch_size=4, language='fa'): # Added language parameter
    # Load all input filepaths
    with open(dataset_manifest, 'r') as f:
        all_filepaths = [json.loads(line) for line in f]

    # Load already transcribed filepaths (if output exists)
    already_transcribed = {}
    if os.path.exists(dataset_manifest_transcribed):
        with open(dataset_manifest_transcribed, 'r') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    logging.warning(f"Skipping empty line at {idx} in {dataset_manifest_transcribed}")
                    continue
                try:
                    entry = json.loads(line)
                    already_transcribed[entry['audio_filepath']] = entry
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping malformed JSON at line {idx} in {dataset_manifest_transcribed}: {e}")

    # Filter out already processed entries
    filepaths_to_process = [entry for entry in all_filepaths if entry['audio_filepath'] not in already_transcribed]

    if not filepaths_to_process:
        logging.info("All files already transcribed.")
        return

    logging.info(f"{len(filepaths_to_process)} files remaining to transcribe.")

    # Keep appending to the output manifest instead of overwriting
    with open(dataset_manifest_transcribed, 'a') as output_file:
        for i in tqdm(range(0, len(filepaths_to_process), batch_size)):
            batch_entries = filepaths_to_process[i:i + batch_size]
            batch_files = [x['audio_filepath'] for x in batch_entries]

            
            transcriptions_batch = []
            for audio_file in batch_files:
                try:
                    segments, _ = model.transcribe(audio_file, beam_size=5, language=language, vad_filter=False) 
                    full_transcription = "".join([segment.text for segment in segments])
                    transcriptions_batch.append(full_transcription)
                except Exception as e:
                    logging.warning(f"Transcription failed for {audio_file}: {e}")
                    transcriptions_batch.append("")  # Empty transcription for failed file
    

            # Write results for the batch
            for idx, transcription in enumerate(transcriptions_batch):
                entry = batch_entries[idx]
                entry['pred_text'] = transcription.strip() # Add strip() for potential leading/trailing spaces
                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                output_file.flush()  # ensure progress is saved in case of interruption


    clean_manifest(dataset_manifest_transcribed)
    calculate_metrics(dataset_manifest_transcribed, edge_len=edge_len)
    logging.info(f"Transcriptions written to {dataset_manifest_transcribed}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe speech using faster-whisper model.")
    parser.add_argument("--model_path", required=True, type=str, help="Path or name of the faster-whisper model (e.g., 'large-v2', 'distil-large-v2').")
    parser.add_argument("--dataset_manifest", required=True, type=str, help="Path to the dataset manifest file.")
    parser.add_argument("--dataset_manifest_transcribed", required=True, type=str, help="Path to save the transcribed manifest file.")
    parser.add_argument("--device", type=int, default=-1, help="Device ID, default is -1 (e.g., CPU)")
    parser.add_argument("--edge_len", type=int, default=7, help="Edge length for start and end CER calculation.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing manifest entries (faster-whisper handles internal batching).")
    parser.add_argument("--compute_type", type=str, default="float16", help="Compute type for faster-whisper ('float16', 'int8_float16', 'int8', 'float32').")
    parser.add_argument("--language", type=str, default="fa", help="Language code for transcription (e.g., 'en', 'fa').")


    args = parser.parse_args()

    # Determine device
    if args.device == -1:
        device = "cpu"
    elif not torch.cuda.is_available():
        raise ValueError(f"CUDA is not available but a GPU device ID ({args.device}) was specified.")
    else:
        device = "cuda"
        device_index = args.device


    # Load model
    model = load_model(args.model_path, device, device_index, args.compute_type)

    logging.info(f"Inference will be done on device: {device}:{device_index} with compute_type: {args.compute_type}")
    # Pass language to transcribe_audio
    transcribe_audio(model, args.dataset_manifest, args.dataset_manifest_transcribed, args.edge_len, batch_size=args.batch_size, language=args.language)


if __name__ == '__main__':
    main()
