# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from scripts.ctc_segmentation.utils import get_segments, get_partitions

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel

parser = argparse.ArgumentParser(description="CTC Segmentation")
parser.add_argument("--output_dir", default="output", type=str, help="Path to output directory")
parser.add_argument(
    "--audio",
    type=str,
    required=True,
    help="Path to directory containing audio files or a single audio file (.wav format)",
)
parser.add_argument(
    "--text",
    type=str,
    required=True,
    help="Path to directory containing transcript files or a single transcript file (.txt format)." 
    "Transcripts must have the same base names as their corresponding audio files.",
)
parser.add_argument("--window_len", type=int, default=8000, help="Window size for ctc segmentation algorithm")
parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate, Hz")
parser.add_argument(
    "--model", type=str, default="QuartzNet15x5Base-En", help="Path to model checkpoint or pre-trained model name",
)
parser.add_argument("--debug", action="store_true", help="Flag to enable debugging messages")
parser.add_argument(
    "--num_jobs",
    default=-2,
    type=int,
    help="The maximum number of concurrently running jobs, `-2` - all CPUs but one are used",
)

logger = logging.getLogger("ctc_segmentation")  # use module name

if __name__ == "__main__":

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # setup logger
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ctc_segmentation_{args.window_len}.log")
    temp_log_file = os.path.join(log_dir, f"ctc_segmentation_temp_{args.window_len}.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    level = "DEBUG" if args.debug else "INFO"

    logger = logging.getLogger("CTC")
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(handlers=handlers, level=level)

    

    audio_path = Path(args.audio)
    text_path = Path(args.text)
    output_dir = Path(args.output_dir)

    if audio_path.is_dir() and text_path.is_dir():
        audio_files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.mp3"))
        # text_files that have the same base name as audio files and exist
        text_files = {f.stem: f for f in text_path.glob("*.txt") if os.path.exists(f)}
        # keep only audio files that have corresponding transcript files
        audio_files = [f for f in audio_files if text_files.get(f.stem)]

    elif audio_path.is_file() and text_path.is_file():
        audio_files = [audio_path]
        text_files = {text_path.stem: text_path}
    else:
        raise ValueError("Both --audio and --text must either point to directories or single files.")

    segments_dir = os.path.join(args.output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    index_duration = None

    batch_size = 100  # Process 100 audio files at a time
    num_batches = (len(audio_files) + batch_size - 1) // batch_size  # Calculate the number of batches

    pbar = tqdm(range(num_batches), desc='Processing batches of audio files')
    for batch_idx in pbar:
        if os.path.exists(args.model):
            asr_model = nemo_asr.models.ASRModel.restore_from(args.model)
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(args.model, strict=False)

        if not (isinstance(asr_model, EncDecCTCModel) or isinstance(asr_model, EncDecHybridRNNTCTCModel)):
            raise NotImplementedError(
                f"Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
                " Currently only instances of these models are supported"
            )

        bpe_model = isinstance(asr_model, nemo_asr.models.EncDecCTCModelBPE) or isinstance(
            asr_model, nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        )

        # get tokenizer used during training, None for char based models
        if bpe_model:
            tokenizer = asr_model.tokenizer
        else:
            tokenizer = None

        if isinstance(asr_model, EncDecHybridRNNTCTCModel):
            asr_model.change_decoding_strategy(decoder_type="ctc")

        # extract ASR vocabulary and add blank symbol
        if hasattr(asr_model, 'tokenizer'):  # i.e. tokenization is BPE-based
            vocabulary = asr_model.tokenizer.vocab
        elif hasattr(asr_model.decoder, "vocabulary"):  # i.e. tokenization is character-based
            vocabulary = asr_model.cfg.decoder.vocabulary
        else:
            raise ValueError("Unexpected model type. Vocabulary list not found.")

        vocabulary = ["ε"] + list(vocabulary)
        logging.debug(f"ASR Model vocabulary: {vocabulary}")
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(audio_files))
        batch_files = audio_files[batch_start:batch_end]

        all_log_probs = []
        all_transcript_file = []
        all_segment_file = []
        all_wav_paths = []

        for ii, path_audio in enumerate(batch_files):
            pbar.set_description(f"({ii}/{batch_size}) Processing {path_audio.name}...")
            transcript_file = text_files.get(path_audio.stem)
            if not transcript_file:
                logging.info(f"No matching transcript found for {path_audio.name}. Skipping.")
                continue
            segment_file = os.path.join(
                segments_dir, f"{args.window_len}_" + path_audio.name.replace(path_audio.suffix, "_segments.txt")
            )
            if not os.path.exists(transcript_file):
                logging.info(f"{transcript_file} not found. Skipping {path_audio.name}")
                continue
            try:
                signal, sample_rate = librosa.load(path_audio, sr=None, mono=True)
                if len(signal) == 0:
                    logging.error(f"Skipping {path_audio.name}")
                    continue

                if sample_rate != args.sample_rate:
                    logging.warning(
                        f"Sampling rate of the audio file {path_audio} ({sample_rate} Hz) does not match "
                        f"--sample_rate={args.sample_rate} Hz. Resampling will occur."
                    )
                    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=args.sample_rate)
                    sample_rate = args.sample_rate

                original_duration = len(signal) / sample_rate
                logging.debug(f"len(signal): {len(signal)}, sr: {sample_rate}")
                logging.debug(f"Duration: {original_duration}s, file_name: {path_audio}")

                # hypotheses = asr_model.transcribe([str(path_audio)], batch_size=1, return_hypotheses=True)
                speech_len = len(signal)
                partitions = get_partitions(t=speech_len, max_len_s=500, fs=sample_rate, samples_to_frames_ratio=1280, overlap=10)

                log_probs = []
                # Process each partition
                for start, end in partitions["partitions"]:
                    audio_chunk = signal[start:end]
                    hypotheses = asr_model.transcribe([audio_chunk], batch_size=1, return_hypotheses=True, verbose=False)
                    # if hypotheses form a tuple (from Hybrid model), extract just "best" hypothesis
                    if type(hypotheses) == tuple and len(hypotheses) == 2:
                        hypotheses = hypotheses[0]
                    
                    chunk_log_probs = hypotheses[0].alignments  # note: "[0]" is for batch dimension unpacking (and here batch size=1)

                    # move blank values to the first column (ctc-package compatibility)
                    blank_col = chunk_log_probs[:, -1].reshape((chunk_log_probs.shape[0], 1))
                    chunk_log_probs  = np.concatenate((blank_col, chunk_log_probs[:, :-1]), axis=1)
                    log_probs.append(chunk_log_probs)

                # Concatenate all partition log_probs and delete overlapping frames
                log_probs = np.vstack(log_probs)
                log_probs = np.delete(log_probs, partitions["delete_overlap_list"], axis=0)


                all_log_probs.append(log_probs)
                all_segment_file.append(str(segment_file))
                all_transcript_file.append(str(transcript_file))
                all_wav_paths.append(path_audio)

                if index_duration is None:
                    index_duration = len(signal) / log_probs.shape[0] / sample_rate

            except Exception as e:
                logging.error(e)
                logging.error(f"Skipping {path_audio.name}")
                continue

        asr_model_type = type(asr_model)
        del asr_model
        torch.cuda.empty_cache()

        if len(all_log_probs) > 0:
            start_time = time.time()

            normalized_lines = []
            for i in tqdm(range(len(all_log_probs)), desc="Running get_segments sequentially"):
                result = get_segments(
                    all_log_probs[i],
                    all_wav_paths[i],
                    all_transcript_file[i],
                    all_segment_file[i],
                    vocabulary,
                    tokenizer,
                    bpe_model,
                    index_duration,
                    args.window_len,
                    log_file=temp_log_file,
                    debug=args.debug,
                )
                with open(temp_log_file, 'r', encoding='utf-8') as f:
                    log_text = f.read()
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(log_text + '\n')
                normalized_lines.append(result)

        total_time = time.time() - start_time
        logger.info(f"Total execution time: ~{round(total_time/60)}min")
        logger.info(f"Saving logs to {log_file}")

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
