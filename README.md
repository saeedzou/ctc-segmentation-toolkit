# CTC Segmentation Toolkit

This tool provides a comprehensive pipeline for aligning long audio files with their corresponding transcripts to generate shorter, segmented audio clips suitable for training Automatic Speech Recognition (ASR) models. It is based on the CTC-Segmentation method, which leverages a pretrained ASR model to find optimal alignments.

More details on the original concept can be found in [this tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb).

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

### 4. Output

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
