# CTC Segmentation Toolkit

This tool provides a comprehensive pipeline for aligning long audio files with their corresponding transcripts to generate shorter, segmented audio clips suitable for training Automatic Speech Recognition (ASR) models. It is based on the [NeMo's dataset creation tool based on CTC-Segmentation](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tools/ctc_segmentation), which leverages a pretrained CTC model to find optimal alignments. It also validates the alignment quality by comparing the alignments with the hypothesis from an ASR.

I extend the toolkit by the following features:

1. The NeMo toolkit cannot process audio files **longer** than a certain duration (depending on the model and hardware, typically around 10 minutes for a 24 GB GPU). This toolkit **allows for the processing of longer audio files** by splitting them into smaller chunks. For most CTC models the `samples_to_frames_ratio` is **1280** (verified for [nvidia/stt_en_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_pc), [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m), and [nvidia/stt_en_fastconformer_transducer_xlarge](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_xlarge)), but you should adjust this value (or the `overlap`) if your model differs — it can be calculated from the model config using the snippet below.

    ```python
    # Get hop length from preprocessor
    hop_length = model.preprocessor.featurizer.hop_length
    sample_rate = model.preprocessor._cfg.sample_rate

    # Get subsampling factor from encoder
    subsampling = model.encoder._cfg.subsampling_factor

    samples_to_frames_ratio = hop_length * subsampling
    frame_shift_ms = samples_to_frames_ratio / sample_rate * 1000

    print("hop_length:", hop_length)
    print("subsampling:", subsampling)
    print("samples_to_frames_ratio:", samples_to_frames_ratio)
    print("frame shift (ms):", frame_shift_ms)
    ```

2. The original toolkit relies on punctuation for segmenting text into utterances, which limits its effectiveness on unpunctuated text sources like YouTube captions or subtitles. Such captions are often short, incomplete sentences, and segmenting them by default can result in abrupt utterances that contain fragments of adjacent sentences. To address this, this toolkit integrates the [Segment Any Text](https://arxiv.org/abs/2406.16678) model, which provides more robust text segmentation. This feature can be enabled by setting `split_using_sat` to `true` in `recipes/config.yaml`.

3. During the preprocessing stage, an additional step is introduced to enhance data quality. The long audio files are first transcribed using a Nemo ASR model, which is selected for its high Real-Time Factor (RTFx), ensuring efficient processing. The Word Error Rate (WER) and Character Error Rate (CER) are then calculated by comparing the ASR-generated hypothesis with the provided transcript. If these error rates exceed the predefined thresholds (`min_wer` and `min_cer` in `recipes/config.yaml`), the corresponding audio-text pair is excluded from further processing steps. This filtering mechanism ensures that only high-quality, accurately transcribed data is used in the subsequent stages of the pipeline.

4. While the original NeMo toolkit validates alignment quality using a single ASR model, this toolkit enhances the validation process by employing multiple ASRs with diverse architectures (including Whisper, NeMo, and wav2vec2). An utterance is only filtered out if it fails to meet the quality thresholds for *all* of the ASR models. This multi-ASR approach leads to a higher data retention rate compared to relying on a single ASR, as the varied strengths of different models ensure that more high-quality segments are correctly identified and retained.


5. This toolkit is highly optimized for the Persian language, featuring a dedicated normalizer located in `preprocessing/normalizer/`. The normalizer's role is to convert text into its spoken form—expanding numbers, dates, and other numerical data—and to remove any out-of-vocabulary characters that are not supported by the NeMo ASR model. The Persian normalizer also includes a feature for transliterating English words and phrases into their Persian equivalents. For other languages, you will need to set the `lang_id` in the configuration file and either provide a custom normalizer or use one of the standard NeMo normalizers.

## Installation

1. **Create a conda environment:**

   ```bash
   conda create -n ctc-segmentation python=3.10.12
   conda activate ctc-segmentation
   ```

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/saeedzou/ctc-segmentation-toolkit.git
    cd ctc-segmentation-toolkit
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install additional dependencies for normalization**:

    ```bash
    git clone https://github.com/saeedzou/ParsNorm.git
    cd ParsNorm && pip install -e . && pip install -r requirements.txt && cd ..
    python -m nltk.downloader -d /usr/nltk_data cmudict
    ```

## Usage

The toolkit operates in three main stages, executed by three corresponding shell scripts. The entire process is configured via a central YAML file.

### 1. Directory Structure

Before running the scripts, organize your data into the following directory structure:

```
<root_dir>/
├── audio/
│   ├── audio_file_1.wav
│   └── audio_file_2.wav
└── text/
    ├── audio_file_1.txt
    └── audio_file_2.txt
```

-   `<root_dir>`: The main directory for your dataset.
-   `audio/`: Contains your long audio files.
-   `text/`: Contains the corresponding verbatim transcripts. Each text file should have the same base name as its audio file.
The text should either have enough end sentence punctuation or be properly segmented into sentences using newlines. Alternatively you can use the `split_using_sat` to segment the text using [Segment Any Text](https://arxiv.org/abs/2406.16678) module.

All processed files and logs will be generated in the `<root_dir>/output/` directory.

### 2. Configuration

All parameters for the pipeline are defined in a YAML file. A template is provided in `recipes/config.yaml`. Before running the scripts, you should customize this file to suit your needs, especially the model paths and processing flags.

### 3. Running the Pipeline

The pipeline is executed by running the three main scripts in order. Each script requires the `<root_dir>` and the path to your configuration file as arguments.

#### **Step 1: Preprocessing**

The preprocessing step includes:

1. Converting the audio to raw `wav` format
2. Normalizing and segmenting the text into utterances for alignment.
3. Transcribing the long audio files to get an initial hypothesis.
4. Calculating error between hypothesis and reference transcripts to ensure verbatim transcription is maintained before alignment. The outliers are removed at this stage.

```bash
bash run_preprocess.sh <root_dir> recipes/config.yaml
```

#### **Step 2: CTC Segmentation**

This is the core step where the CTC-based alignment is performed. It runs the segmentation, verifies the results, and cuts the audio into smaller segments based on the alignments. It generates a manifest `json` file which contains information about the segmented audio files and their corresponding transcripts.

```bash
bash run_ctc_segmentation.sh <root_dir> recipes/config.yaml
```

#### **Step 3: ASR Transcription**

After segmentation, this script runs multiple ASR models on the newly created short audio segments to generate high-quality transcriptions, calculate various metrics including CER, WER, insertion, deletion, substitution rates, edge CER's for start and end of the transcripts.

```bash
bash run_asr.sh <root_dir> recipes/config.yaml
```

### 4. Merging and Filtering Manifests

After the ASR transcription step, you may want to merge the individual manifests generated by each model into a single consolidated manifest. This can be done using the `merge_manifests.py` script.

```bash
python -m scripts.post_processing.merge_manifests --output_dir <root_dir>/output --model_names nemo wav2vec2_v3 faster_whisper_large
```

Then you can filter the merged manifest using the `filter_samples.py` script.

```bash
python -m scripts.post_processing.filter_samples --config_path recipes/config.yaml --manifest_path <root_dir>/output/manifests/merged_manifest.json
```

### 5. Output

The final output will be located in `<root_dir>/output/`. Key outputs include:

- `manifests/`: Contains the final JSON manifests with audio paths, durations, and transcriptions.
- `segments/`: The audio segmenting information from different alignment windows (start, end times and alignment scores).
- `verified_segments/`: The verified audio segments that have high alignment confidence scores.
- `logs/`: Detailed logs for each step of the process.
- `clips_16k/`: The 16kHz audio segments extracted from the segment files in `verified_segments/`.

By following these steps, you can efficiently process a large corpus of long audio and text into a high-quality dataset for training ASR models.

## Acknowledgement

This tool is based on the [CTC Segmentation](https://github.com/lumaku/ctc-segmentation):
**CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition**
https://doi.org/10.1007/978-3-030-60276-5_27 or pre-print https://arxiv.org/abs/2007.09127

```
@InProceedings{ctcsegmentation,
author="K{\"u}rzinger, Ludwig
and Winkelbauer, Dominik
and Li, Lujun
and Watzel, Tobias
and Rigoll, Gerhard",
editor="Karpov, Alexey
and Potapova, Rodmonga",
title="CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition",
booktitle="Speech and Computer",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="267--278",
abstract="Recent end-to-end Automatic Speech Recognition (ASR) systems demonstrated the ability to outperform conventional hybrid DNN/HMM ASR. Aside from architectural improvements in those systems, those models grew in terms of depth, parameters and model capacity. However, these models also require more training data to achieve comparable performance.",
isbn="978-3-030-60276-5"
}
```
