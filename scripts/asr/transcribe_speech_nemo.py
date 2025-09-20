import argparse
import json
import os
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.utils import logging
from tqdm import tqdm
from scripts.preprocessing.normalizer.calculate_metrics import calculate_metrics

def load_model(model_path):
    """Loads a NeMo ASR model."""
    if model_path.endswith('.nemo'):
        model = ASRModel.restore_from(model_path)
    else:
        model = ASRModel.from_pretrained(model_path)
    return model

def transcribe_audio(model, dataset_manifest, dataset_manifest_transcribed, edge_len, batch_size=1):
    """Handles the transcription process using a NeMo ASR model."""
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
        audio_filepaths = [entry['audio_filepath'] for entry in filepaths_to_process]
        
        for i in tqdm(range(0, len(audio_filepaths), batch_size)):
            try:
                batch_entries = filepaths_to_process[i:i + batch_size]
                batch_files = [x['audio_filepath'] for x in batch_entries]

                transcriptions_batch = model.transcribe(
                    audio=batch_files,
                    batch_size=batch_size,
                    verbose=False
                )

            except Exception as e:
                logging.warning(f"Transcription failed for a batch starting with {batch_files[0]}: {e}")
                transcriptions_batch = [""] * len(batch_files) # Empty transcriptions for the failed batch

            for idx, transcription in enumerate(transcriptions_batch):
                entry = batch_entries[idx]
                entry['pred_text'] = transcription.text
                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                output_file.flush()  # ensure progress is saved in case of interruption

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
    clean_manifest(dataset_manifest_transcribed)
    calculate_metrics(dataset_manifest_transcribed, edge_len=edge_len)

    logging.info(f"Transcriptions written to {dataset_manifest_transcribed}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe speech using a NeMo ASR model.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the pretrained NeMo model (.nemo file or pretrained name).")
    parser.add_argument("--dataset_manifest", required=True, type=str, help="Path to the dataset manifest file.")
    parser.add_argument("--dataset_manifest_transcribed", required=True, type=str, help="Path to save the transcribed manifest file.")
    parser.add_argument("--device", type=int, default=-1, help="Device ID, default is -1 (e.g., CPU)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for transcription.")
    parser.add_argument("--edge_len", type=int, default=7, help="Edge length for start and end CER calculation.")
    args = parser.parse_args()

    model = load_model(args.model_path)
    
    if args.device == -1:
        device = torch.device("cpu")
        model = model.to(device)
    elif torch.cuda.is_available() and args.device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.device}")
        model = model.to(device)
    else:
        raise ValueError(f"Invalid device ID {args.device}. CUDA not available or index out of range.")

    model.eval()

    logging.info(f"Inference will be done on device: {device}")
    transcribe_audio(model, args.dataset_manifest, args.dataset_manifest_transcribed, args.edge_len, args.batch_size)

if __name__ == '__main__':
    main()
