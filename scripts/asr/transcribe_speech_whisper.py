import argparse
import json
import torch
import os
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.utils import logging
from tqdm import tqdm
from scripts.preprocessing.normalizer.calculate_metrics import calculate_metrics


# Load Whisper model and processor
def load_model_and_processor(model_path, processor_path):
    model = WhisperForConditionalGeneration.from_pretrained(model_path, use_cache=False)
    processor = WhisperProcessor.from_pretrained(processor_path, language='fa', task='transcribe')
    return model, processor

# This function processes a batch of audio files in parallel, improving efficiency
def process_audio_files_batch(batch_files, processor):
    input_features_list = []
    attention_mask_list = []

    for audio_file in batch_files:
        audio_data, sampling_rate = torchaudio.load(audio_file, normalize=True)
        audio_data = audio_data.mean(dim=0)  # Convert stereo to mono if necessary
        inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")

        input_features_list.append(inputs.input_features)

        # If attention_mask exists, use it; otherwise, create one
        if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
            attention_mask_list.append(inputs.attention_mask)
        else:
            # Create a dummy attention mask of ones
            dummy_mask = torch.ones(inputs.input_features.shape[:-1], dtype=torch.long)
            attention_mask_list.append(dummy_mask)

    input_features_batch = torch.cat(input_features_list, dim=0)
    attention_mask_batch = torch.cat(attention_mask_list, dim=0)

    return input_features_batch, attention_mask_batch



def transcribe_audio(model, processor, device, dataset_manifest, dataset_manifest_transcribed, edge_len, batch_size=4):
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

            audio_input, attention_mask = process_audio_files_batch(batch_files, processor)
            audio_input = audio_input.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                generated_ids = model.generate(audio_input, attention_mask=attention_mask)
                transcriptions_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, transcription in enumerate(transcriptions_batch):
                entry = batch_entries[idx]
                entry['pred_text'] = transcription
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
    parser = argparse.ArgumentParser(description="Transcribe speech using Whisper model.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the pretrained Whisper model.")
    parser.add_argument("--processor_path", required=True, type=str, help="Path to the Whisper processor.")
    parser.add_argument("--dataset_manifest", required=True, type=str, help="Path to the dataset manifest file.")
    parser.add_argument("--dataset_manifest_transcribed", required=True, type=str, help="Path to save the transcribed manifest file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--edge_len", type=int, default=7, help="Edge length for start and end CER calculation.")
    parser.add_argument("--device", type=int, default=-1, help="Device ID, default is -1 (e.g., CPU)")
    
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_path, args.processor_path)
    if args.device == -1:
        device = torch.device("cpu")
    elif torch.cuda.is_available() and args.device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.device}")
    else:
        raise ValueError(f"Invalid device ID {args.device}. CUDA not available or index out of range.")
    model = model.to(device)

    logging.info(f"Inference will be done on device: {device}")
    transcribe_audio(model, processor, device, args.dataset_manifest, args.dataset_manifest_transcribed, args.edge_len, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
