import os
import re
import argparse
import librosa
import torch
from nemo.collections.asr.models import ASRModel

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    # convert to mono
    waveform = librosa.to_mono(waveform)
    # Normalize and convert to float32
    if waveform.dtype == 'int16':
        waveform = waveform.astype('float32') / 32768.0
    elif waveform.dtype == 'int32':
        waveform = waveform.astype('float32') / 2147483648.0
    elif waveform.dtype == 'uint8':
        waveform = (waveform.astype('float32') - 128) / 128.0
    else:
        waveform = waveform.astype('float32')
    return waveform, sample_rate

def chunk_audio(waveform, sample_rate, chunk_length=30):
    chunk_indices = []
    for i in range(0, len(waveform), chunk_length * sample_rate):
        start_idx = i
        end_idx = min(i + chunk_length * sample_rate, len(waveform))
        chunk_indices.append((start_idx, end_idx))
    return chunk_indices

def transcribe_chunk_batch(chunks, model, batch_size):
    transcriptions = model.transcribe(chunks, batch_size=min(batch_size, len(chunks)), verbose=False, return_hypotheses=False)
    return [t.text for t in transcriptions]

MIN_SAMPLES = 512

def transcribe_audio(file_path, model, batch_size):
    waveform, sample_rate = load_audio(file_path)
    chunk_indices = chunk_audio(waveform, sample_rate)
    transcriptions = []
    batch = []

    for start, end in chunk_indices:
        chunk = waveform[start:end]
        if len(chunk) < MIN_SAMPLES:
            continue  # Skip too-short chunks
        batch.append(chunk)
        if len(batch) == batch_size:
            transcriptions.extend(transcribe_chunk_batch(batch, model, batch_size))
            batch = []

    if batch:
        transcriptions.extend(transcribe_chunk_batch(batch, model, batch_size))

    transcriptions = ' '.join(transcriptions)
    transcriptions = re.sub(r'\s+', ' ', transcriptions)
    return transcriptions.strip()

def load_model(model_path, device):
    model = ASRModel.restore_from(restore_path=model_path)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Transcribe all .wav files in a directory using NeMo ASR model.")
    parser.add_argument('--model_path', required=True, help='Path to pre-trained ASR model')
    parser.add_argument('--audio_dir', required=True, help='Directory containing .wav files')
    parser.add_argument('--transcript_dir', required=True, help='Directory to save transcripts')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for transcribing audio chunks')
    parser.add_argument('--device', type=int, default=-1, help='CUDA device index or -1 for auto (CPU if no GPU)')
    args = parser.parse_args()

    # Determine device
    if args.device == -1:
        device = torch.device("cpu")
    elif torch.cuda.is_available() and args.device < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.device}")
    else:
        raise ValueError(f"Invalid device ID {args.device}. CUDA not available or index out of range.")

    os.makedirs(args.transcript_dir, exist_ok=True)
    model = load_model(args.model_path, device)

    audio_files = [f for f in os.listdir(args.audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        print(f"No .wav or .mp3 files found in {args.audio_dir}.")
        return

    for filename in audio_files:
        audio_path = os.path.join(args.audio_dir, filename)
        transcript_path = os.path.join(args.transcript_dir, os.path.splitext(filename)[0] + '.txt')
        print(f"Transcribing {filename}...")
        if os.path.exists(transcript_path):
            print(f"Transcription is already done for {filename} at {transcript_path}")
            continue
        transcription = transcribe_audio(audio_path, model, args.batch_size)
        with open(transcript_path, 'w') as f:
            f.write(transcription)
        print(f"Saved transcript to {transcript_path}")

    print("Long speech transcription done!")

if __name__ == "__main__":
    main()
