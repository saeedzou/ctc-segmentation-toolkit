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
import os
import re
from glob import glob
from typing import List, Optional

import numpy as np
import regex
import soundfile as sf
import torch
from num2words import num2words
from pydub import AudioSegment
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.utils import model_utils

from scripts.preprocessing.normalizer.normalization_helpers import LATIN_TO_RU, RU_ABBREVIATIONS


try:
    from parsnorm import ParsNorm
    PARSNORM_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PARSNORM_AVAILABLE = False


try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    NEMO_NORMALIZATION_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    NEMO_NORMALIZATION_AVAILABLE = False


parser = argparse.ArgumentParser(description="Prepares text and audio files for segmentation")
parser.add_argument("--in_text", type=str, default=None, help="Path to a text file or a directory with .txt files")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument("--audio_dir", type=str, help="Path to folder with .mp3 or .wav audio files")
parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate used during ASR model training, Hz")
parser.add_argument("--bit_depth", type=int, default=16, help="Bit depth to use for processed audio files")
parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
parser.add_argument(
    "--language",
    type=str,
    default="en",
    choices=["en", "ru", "de", "es", "fa", 'other'],
    help='Add target language based on the num2words list of supported languages',
)
parser.add_argument(
    "--cut_prefix", type=int, default=0, help="Number of seconds to cut from the beginning of the audio files.",
)
parser.add_argument(
    "--model", type=str, default="QuartzNet15x5Base-En", help="Pre-trained model name or path to model checkpoint"
)
parser.add_argument(
    "--max_length", type=int, default=40, help="Max number of words of the text segment for alignment."
)
parser.add_argument(
    "--additional_split_symbols",
    type=str,
    default="",
    help="Additional symbols to use for \
    sentence split if eos sentence split resulted in sequence longer than --max_length. "
    "Use '|' as a separator between symbols, for example: ';|:'. Use '\\s' to split by space.",
    )
parser.add_argument(
    "--split_using_sat", type=lambda x: x.lower() == 'true', default=False, help="Whether to use SaT for text segmentation"
)
parser.add_argument(
    "--sat_model", type=str, default="sat-12l-sm", help="SaT checkpoint path"
)
parser.add_argument(
    "--remove_brackets", type=lambda x: x.lower() == 'true', default=True, help="Whether to remove text in square brackets or not"
)
parser.add_argument(
    "--remove_number_in_brackets", type=lambda x: x.lower() == 'true', default=True, help="Whether to remove numbers in square brackets or not"
)
parser.add_argument(
    "--remove_curly_brackets", type=lambda x: x.lower() == 'true', default=False, help="Whether to remove text in curly brackets or not"
)
parser.add_argument(
    "--remove_number_in_curly_brackets", type=lambda x: x.lower() == 'true', default=False, help="Whether to remove numbers in curly brackets or not"
)
parser.add_argument(
    "--remove_asterisks", type=lambda x: x.lower() == 'true', default=False, help="Whether to remove text in asterisks or not"
)
parser.add_argument(
    "--remove_parentheses", type=lambda x: x.lower() == 'true', default=False, help="Whether to remove text in parentheses or not"
)
parser.add_argument(
    "--remove_number_in_parentheses", type=lambda x: x.lower() == 'true', default=False, help="Whether to remove numbers in parentheses or not"
)
parser.add_argument(
    "--remove_number_in_guillumets", type=lambda x: x.lower() == 'true', default=False, help="Whether to remove numbers in guillumets or not"
)
parser.add_argument(
    "--remove_speaker_labels", type=lambda x: x.lower() == 'true', default=False, 
    help="Whether to remove reported speech or not. For example, 'John: Hello, how are you?' \
    will be converted to 'Hello, how are you?'"
)
parser.add_argument(
    "--split_using_pattern",
    type=lambda x: x.lower() == 'true',
    default=True,
    help="Whether to split text using the defined regex pattern for sentence boundary detection or not",
)
parser.add_argument(
    "--split_on_quotes", type=lambda x: x.lower() == 'true', default=False, help="Whether to split on quotes or not. «» is used for Persian"
)
parser.add_argument(
    "--split_on_verbs", type=lambda x: x.lower() == 'true', default=True, help="Whether to split more on verbs or not"
)
parser.add_argument(
    "--pos_tagger_path", type=str, default='pos_tagger.model', help="Path to the POS tagger model."
)
parser.add_argument(
    "--split_on_verbs_min_words", type=int, default=5, help="The minimum number of words available for the sentence to be split"
)
parser.add_argument(
    "--split_on_verbs_max_words", type=int, default=20, help="The word threshold to run splitting on verbs"
)
parser.add_argument(
    "--use_nemo_normalization",
    action="store_true",
    help="Set to True to use NeMo Normalization tool to convert numbers from written to spoken format.",
)
parser.add_argument(
    "--batch_size", type=int, default=100, help="Batch size for NeMo Normalization tool.",
)
parser.add_argument(
    "--tqdm", type=str, default='default', help="whether to use tqdm for notebook or default for terminal. can be notebook or default",
)




def process_audio(
    in_file: str, wav_file: str = None, cut_prefix: int = 0, sample_rate: int = 16000, bit_depth: int = 16
):
    """Process audio file: format conversion, resampling, channel reduction, and prefix cutting using pydub.

    Args:
        in_file: path to the input audio file (any format supported by ffmpeg)
        wav_file: path to the output .wav file
        cut_prefix: number of seconds to cut from the beginning of the audio file
        sample_rate: target sampling rate in Hz
        bit_depth: target bit depth (sample width in bytes for pydub)
    """
    try:
        if not os.path.exists(in_file):
            raise ValueError(f'{in_file} not found')
        if wav_file is None:
            raise ValueError('Output wav_file path must be provided')
        if os.path.exists(wav_file):
            print(f'{wav_file} already exists. Skipping processing.')
            return  # Skip processing

        # Ensure output directory exists
        os.makedirs(os.path.dirname(wav_file), exist_ok=True)

        # Load audio file (pydub handles mp3 etc.)
        audio = AudioSegment.from_file(in_file)

        # Cut prefix if needed
        if cut_prefix > 0:
            cut_prefix_ms = cut_prefix * 1000  # milliseconds
            audio = audio[cut_prefix_ms:]

        # Set sample rate and convert to mono
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(1)

        # Set sample width (bit depth)
        sample_width_bytes = bit_depth // 8
        audio = audio.set_sample_width(sample_width_bytes)

        # Convert pydub AudioSegment to NumPy array
        samples = np.array(audio.get_array_of_samples())

        # Normalize to float or keep int depending on bit depth
        if bit_depth == 16:
            dtype = 'int16'
            subtype = 'PCM_16'
        elif bit_depth == 24:
            dtype = 'int32'  # pydub stores 24-bit as int32
            subtype = 'PCM_24'
        else:
            raise ValueError(f'Unsupported bit depth: {bit_depth}. Only 16 or 24 bits supported.')

        # Write WAV with soundfile
        sf.write(wav_file, samples.astype(dtype), samplerate=sample_rate, subtype=subtype)

        # Optional success message
        # print(f'Successfully processed {in_file} to {wav_file}')

    except Exception as e:
        print(f'{in_file} skipped - An error occurred during processing: {e}')


def split_text(
    in_file: str,
    out_file: str,
    vocabulary: List[str],
    pos_tagger,
    sat,
    language="en",
    remove_brackets: bool = True,
    remove_curly_brackets: bool = False,
    remove_asterisks: bool = False,
    remove_parentheses: bool = False,
    remove_number_in_brackets: bool = False,
    remove_number_in_curly_brackets: bool = False,
    remove_number_in_parentheses: bool = False,
    remove_number_in_guillumets: bool = False,
    remove_speaker_labels: bool = False,
    do_lower_case: bool = True,
    max_length: bool = 100,
    additional_split_symbols: str = None,
    split_using_pattern: bool = True,
    split_on_quotes: bool = False,
    split_on_verbs: bool = True,
    split_on_verbs_min_words: int = 5,
    split_on_verbs_max_words: int = 15,
    use_nemo_normalization: bool = False,
    n_jobs: Optional[int] = 1,
    batch_size: Optional[int] = 1.0,
):
    """
    Breaks down the in_file roughly into sentences. Each sentence will be on a separate line.
    Written form of the numbers will be converted to its spoken equivalent, OOV punctuation will be removed.

    Args:
        in_file: path to original transcript
        out_file: path to the output file
        vocabulary: ASR model vocabulary
        language: text language
        remove_brackets: Set to True if square [] and curly {} brackets should be removed from text.
            Text in square/curly brackets often contains inaudible fragments like notes or translations
        remove_asterisks: Set to True if text in asterisks should be removed from text.
            Text in asterisks often contains inaudible fragments like notes or translations
        remove_parentheses: Set to True if text in parentheses should be removed from text.
            Text in parentheses often contains inaudible fragments like notes or translations
        do_lower_case: flag that determines whether to apply lower case to the in_file text
        max_length: Max number of words of the text segment for alignment
        split_using_pattern: Set to True to split text using the defined regex pattern for sentence boundary detection,
            otherwise split based on new lines.
        additional_split_symbols: Additional symbols to use for sentence split if eos sentence split resulted in
            segments longer than --max_length
        use_nemo_normalization: Set to True to use NeMo normalization tool to convert numbers from written to spoken
            format. Normalization using num2words will be applied afterwards to make sure there are no numbers present
            in the text, otherwise they will be replaced with a space and that could deteriorate segmentation results.
        n_jobs (if use_nemo_normalization=True): the maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given,
                no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
        batch_size (if use_nemo_normalization=True): Number of examples for each process
    """
    with open(in_file, "r") as f:
        transcript = f.read()
    transcript = re.sub(r'^\s*\n', '', transcript, flags=re.MULTILINE)
    # remove some symbols for better split into sentences
    transcript = (
        transcript.replace("\t", " ")
        .replace("…", "...")
        .replace("\\", " ")
        .replace("--", " -- ")
        .replace(". . .", "...")
        .replace("﴿", "«")
        .replace("﴾", "»")
    )

    # Replace ٪ with % for Persian
    if language == 'fa':
        transcript = transcript.replace("٪", "%")

    # end of quoted speech - to be able to split sentences by full stop
    if language == 'fa':
        transcript = re.sub(r"([\.\?\!\؟])([\"\'”«»])", r"\g<2>\g<1> ", transcript)
    else:
        transcript = re.sub(r"([\.\?\!])([\"\'”])", r"\g<2>\g<1> ", transcript)

    # remove extra space
    transcript = re.sub(r" +", " ", transcript)

    if remove_brackets:
        transcript = re.sub(r'(\[.*?\])', ' ', transcript)
    if remove_curly_brackets:
        transcript = re.sub(r'(\{.*?\})', ' ', transcript)
    if remove_asterisks:
        transcript = re.sub(r'(\*.*?\*)', ' ', transcript)
    if remove_parentheses:
        transcript = re.sub(r'(\(.*?\))', ' ', transcript)
    
    if remove_number_in_brackets:
        transcript = re.sub(r'(\[\s*\d+\s*\])', ' ', transcript)
    if remove_number_in_curly_brackets:
        transcript = re.sub(r'(\{\s*\d+\s*\})', ' ', transcript)
    if remove_number_in_parentheses:
        transcript = re.sub(r'(\(\s*\d+\s*\))', ' ', transcript)
    if remove_number_in_guillumets:
        transcript = re.sub(r'«\s*\d+\s*»', ' ', transcript)

    if remove_speaker_labels:
        transcript = re.sub(r'^\s*\w+\s*: ', '', transcript, flags=re.MULTILINE)
        transcript = re.sub(r'^\s*[\u0600-\u06FF]+(?:\s*):', '', transcript, flags=re.MULTILINE)
        transcript = re.sub(r'^\s*-\s*\w+\s*:', '', transcript, flags=re.MULTILINE)
        transcript = re.sub(r'^\s*-\s*[\u0600-\u06FF]+(?:\s*):', '', transcript, flags=re.MULTILINE)
        transcript = re.sub(r'^\s*\+\s*\w+\s*:', '', transcript, flags=re.MULTILINE)
        transcript = re.sub(r'^\s*\+[\u0600-\u06FF]+(?:\s*):', '', transcript, flags=re.MULTILINE)
    
    lower_case_unicode = ''
    upper_case_unicode = ''
    if language == "ru":
        lower_case_unicode = '\u0430-\u04FF'
        upper_case_unicode = '\u0410-\u042F'
    elif language == "fa":
        lower_case_unicode = '\u0600-\u06FF'  # Arabic and Persian characters
        upper_case_unicode = ''  # Persian doesn't have uppercase letters
    elif language not in ["ru", "en"]:
        print(f"Consider using {language} unicode letters for better sentence split.")

    # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
    matches = re.findall(r'[a-z' + lower_case_unicode + r']\.\s[a-z' + lower_case_unicode + r']\.', transcript)
    for match in matches:
        transcript = transcript.replace(match, match.replace('. ', '.'))

    # find phrases in quotes
    if split_on_quotes:
        if language == 'fa':
            with_quotes = re.finditer(r'«[^»]+»', transcript)
        else:
            with_quotes = re.finditer(r'“[A-Za-z ?]+.*?”', transcript)
        sentences = []
        last_idx = 0
        for m in with_quotes:
            match = m.group()
            match_idx = m.start()
            if last_idx < match_idx:
                sentences.append(transcript[last_idx:match_idx])
            sentences.append(match)
            last_idx = m.end()
        sentences.append(transcript[last_idx:])
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        sentences = [transcript]


    # Read and split transcript by utterance (roughly, sentences)
    if language == 'fa':
        split_pattern = r"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]\.)(?<![A-Z{upper_case_unicode}]\.)(?<=\.)\s"
    else:
        split_pattern = r"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]\.)(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s"
    
    if split_using_pattern:
        # Split using the defined regex pattern for sentence boundary detection
        sentences = [s.strip() for sent in sentences for s in regex.split(split_pattern, sent) for s in s.split("\n") if s.strip()]
    else:
        # Split based on new lines if the flag is False
        sentences = [s.strip() for sent in sentences for s in sent.split("\n") if s.strip()]

    def additional_split(sentences, split_on_symbols):
        if len(split_on_symbols) == 0:
            return sentences

        split_on_symbols = split_on_symbols.split("|")

        def _split(sentences, delimiter):
            result = []
            for sent in sentences:
                split_sent = sent.split(delimiter)
                # keep the delimiter
                split_sent = [(s + delimiter).strip() for s in split_sent[:-1]] + [split_sent[-1]]

                if "," in delimiter:
                    # split based on comma usually results in too short utterance, combine sentences
                    # that result in a single word split. It's usually not recommended to do that for other delimiters.
                    comb = []
                    for s in split_sent:
                        MIN_LEN = 2
                        # if the previous sentence is too short, combine it with the current sentence
                        if len(comb) > 0 and (len(comb[-1].split()) <= MIN_LEN or len(s.split()) <= MIN_LEN):
                            comb[-1] = comb[-1] + " " + s
                        else:
                            comb.append(s)
                    result.extend(comb)
                else:
                    result.extend(split_sent)
            return result

        another_sent_split = []
        for sent in sentences:
            split_sent = [sent]
            for delimiter in split_on_symbols:
                if len(delimiter) == 0:
                    continue
                split_sent = _split(split_sent, delimiter + " " if delimiter != " " else delimiter)
            another_sent_split.extend(split_sent)

        sentences = [s.strip() for s in another_sent_split if s.strip()]
        return sentences
    if split_using_pattern and additional_split_symbols:
        additional_split_symbols = additional_split_symbols.replace("/s", " ")
        sentences = additional_split(sentences, additional_split_symbols)

    vocabulary_symbols = []
    for x in vocabulary:
        if x != "<unk>":
            # for BPE models
            vocabulary_symbols.extend([x for x in x.replace("##", "").replace("▁", "")])
    vocabulary_symbols = list(set(vocabulary_symbols))
    vocabulary_symbols += [x.upper() for x in vocabulary_symbols]

    # check to make sure there will be no utterances for segmentation with only OOV symbols
    vocab_no_space_with_digits = set(vocabulary_symbols + [str(i) for i in range(10)])
    if " " in vocab_no_space_with_digits:
        vocab_no_space_with_digits.remove(" ")


    if split_on_verbs:
        from hazm import word_tokenize

        def split_sentence_by_verbs(text, tagger, word_min_threshold):
            tokens = word_tokenize(text)
            tagged_tokens = tagger.tag(tokens)

            verb_indices = [
                i + 1 if (i + 1 < len(tagged_tokens) and tagged_tokens[i + 1][1] == 'PUNCT') else i
                for i, (_, pos) in enumerate(tagged_tokens)
                if pos == 'VERB'
            ]

            if not verb_indices:
                return [text]

            def clean_token(token, pos):
                return token.replace('_', ' ').strip() if pos != 'PUNCT' else token

            start_idx = 0
            result = []
            for idx in verb_indices:
                if idx - start_idx > word_min_threshold:
                    split = tagged_tokens[start_idx:idx+1]
                    cleaned = [clean_token(tok, pos) for tok, pos in split]
                    result.append(" ".join(cleaned))
                    start_idx = idx + 1

            if start_idx < len(tagged_tokens):
                last_chunk = tagged_tokens[start_idx:]
                cleaned = [clean_token(tok, pos) for tok, pos in last_chunk]
                if len(last_chunk) < word_min_threshold and result:
                    result[-1] += " " + " ".join(cleaned)
                else:
                    result.append(" ".join(cleaned))

            return result

        new_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > split_on_verbs_max_words:
                new_sentences.extend(split_sentence_by_verbs(sentence, pos_tagger, split_on_verbs_min_words))
            else:
                new_sentences.append(sentence)
        sentences = [s.strip() for s in new_sentences if s.strip()]

    processed_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > max_length:
            current_sub_sentences = [sentence]
            if sat is not None:
                sat_split_result = sat.split(sentence)
                # Check if SAT actually split it or made it shorter
                if len(sat_split_result) > 1 or (len(sat_split_result) == 1 and len(sat_split_result[0].split()) < len(sentence.split())):
                    current_sub_sentences = sat_split_result

            final_chunks_for_sentence = []
            for sub_sent in current_sub_sentences:
                if len(sub_sent.split()) > max_length:
                    # Apply max_length word split
                    words = sub_sent.split()
                    for i in range(0, len(words), max_length):
                        final_chunks_for_sentence.append(" ".join(words[i : i + max_length]))
                else:
                    final_chunks_for_sentence.append(sub_sent)
            processed_sentences.extend([s.strip() for s in final_chunks_for_sentence if s.strip()])
        else:
            processed_sentences.append(sentence.strip())

    sentences = [s for s in processed_sentences if s] # Filter out any empty strings
    # save split text with original punctuation and case
    if language == "fa":
        if not PARSNORM_AVAILABLE:
            raise ModuleNotFoundError(
                "ParsNorm not installed. Please install it for Persian text processing `pip install parsnorm`"
            )
        normalizer = ParsNorm()
        
        new_sentences = []
        for sentence in sentences:
            sentence = normalizer.normalize(sentence,
                                            repeated_punctuation_removal=True,
                                            symbol_pronounciation=True,
                                            en_fa_transliteration=True,
                                            arabic_correction=True, 
                                            special_chars_removal=True, 
                                            number_correction=True, 
                                            punctuation_correction=False, 
                                            comma_between_numbers_removal=False, 
                                            english_correction=True,
                                            convert_time=False, 
                                            convert_date=False,
                                            alphabet_correction=False, 
                                            semi_space_correction=False,
                                            date_abbrev_replacement=False, 
                                            persian_label_abbrev_replacement=False,
                                            law_abbrev_replacement=False, 
                                            book_abbrev_replacement=False, 
                                            other_abbrev_replacement=False, 
                                            number_conversion=False, 
                                            hazm=False,
                                            remove_punct=False,
                                            keep_allowed_chars=False)
            new_sentences.append(sentence)
    else:
        new_sentences = sentences
    
    sentences = [
        s.strip() for s in new_sentences if len(vocab_no_space_with_digits.intersection(set(s.lower()))) > 0 and s.strip()
    ]
    
    out_dir, out_file_name = os.path.split(out_file)
    with open(os.path.join(out_dir, out_file_name[:-4] + "_with_punct.txt"), "w") as f:
        f.write(re.sub(r' +', ' ', "\n".join(sentences)))

    # substitute common abbreviations before applying lower case
    if language == "ru":
        for k, v in RU_ABBREVIATIONS.items():
            sentences = [s.replace(k, v) for s in sentences]
        # replace Latin characters with Russian
        for k, v in LATIN_TO_RU.items():
            sentences = [s.replace(k, v) for s in sentences]

    if language == "en" and use_nemo_normalization:
        if not NEMO_NORMALIZATION_AVAILABLE:
            raise ValueError("NeMo normalization tool is not installed.")

        print("Using NeMo normalization tool...")
        normalizer = Normalizer(input_case="cased", cache_dir=os.path.join(os.path.dirname(out_file), "en_grammars"))
        sentences_norm = normalizer.normalize_list(
            sentences, verbose=False, punct_post_process=True, n_jobs=n_jobs, batch_size=batch_size
        )
        if len(sentences_norm) != len(sentences):
            raise ValueError("Normalization failed, number of sentences does not match.")
        else:
            sentences = sentences_norm

    if language == "fa":
        sentences = [normalizer.normalize(s,
                                          punctuation_correction=True, 
                                          comma_between_numbers_removal=True, 
                                          convert_time=True, 
                                          convert_date=True,
                                          alphabet_correction=True, 
                                          semi_space_correction=True,
                                          date_abbrev_replacement=True,
                                          persian_label_abbrev_replacement=True,
                                          law_abbrev_replacement=True, 
                                          book_abbrev_replacement=True, 
                                          other_abbrev_replacement=True, 
                                          number_conversion=True, 
                                          hazm=True,
                                          remove_punct=True,
                                          keep_allowed_chars=True,
                                          english_correction=False,
                                          repeated_punctuation_removal=False,
                                          symbol_pronounciation=False,
                                          en_fa_transliteration=False,
                                          arabic_correction=False, 
                                          special_chars_removal=False, 
                                          number_correction=False)
                        for s in sentences]

    sentences = '\n'.join(sentences)

    # replace numbers with num2words
    try:
        p = re.compile(r"\d+")
        new_text = ""
        match_end = 0
        for i, m in enumerate(p.finditer(sentences)):
            match = m.group()
            match_start = m.start()
            if i == 0:
                new_text = sentences[:match_start]
            else:
                new_text += sentences[match_end:match_start]
            match_end = m.end()
            new_text += sentences[match_start:match_end].replace(match, num2words(match, lang=language))
        new_text += sentences[match_end:]
        sentences = new_text
    except NotImplementedError:
        print(
            f"{language} might be missing in 'num2words' package. Add required language to the choices for the"
            f"--language argument."
        )
        raise

    sentences = re.sub(r' +', ' ', sentences)

    with open(os.path.join(out_dir, out_file_name[:-4] + "_with_punct_normalized.txt"), "w") as f:
        f.write(sentences)

    if do_lower_case:
        sentences = sentences.lower()
    if language == 'fa':
        sentences = sentences.replace("ۀ", "ه ی")
    symbols_to_remove = ''.join(set(sentences).difference(set(vocabulary_symbols + ["\n", " "])))
    sentences = sentences.translate(''.maketrans(symbols_to_remove, len(symbols_to_remove) * " "))

    # remove extra space
    sentences = re.sub(r' +', ' ', sentences)
    with open(out_file, "w") as f:
        f.write(sentences)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    text_files = []
    if args.in_text:
        if args.split_on_verbs:
            from hazm import POSTagger

            tagger = POSTagger(model=args.pos_tagger_path)
        else:
            tagger = None

        if args.split_using_sat:
            from wtpsplit import SaT

            sat = SaT(args.sat_model, ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            sat = None

        if args.model is None:
            raise ValueError(f"ASR model must be provided to extract vocabulary for text processing")
        elif os.path.exists(args.model):
            print(f"Restoring model from {args.model}")
            model_cfg = ASRModel.restore_from(restore_path=args.model, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            print(f"Restoring model : {imported_class.__name__}")
            asr_model = imported_class.restore_from(restore_path=args.model)  # type: ASRModel
            model_name = os.path.splitext(os.path.basename(args.model))[0]
        else:
            # restore model by name
            asr_model = ASRModel.from_pretrained(model_name=args.model)  # type: ASRModel
            model_name = args.model

        if not (isinstance(asr_model, EncDecCTCModel) or isinstance(asr_model, EncDecHybridRNNTCTCModel)):
            raise NotImplementedError(
                f"Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
                " Currently only instances of these models are supported"
            )

        # get vocabulary list
        if hasattr(asr_model, 'tokenizer'):  # i.e. tokenization is BPE-based
            vocabulary = asr_model.tokenizer.vocab
        elif hasattr(asr_model.decoder, "vocabulary"):  # i.e. tokenization is character-based
            vocabulary = asr_model.cfg.decoder.vocabulary
        else:
            raise ValueError("Unexpected model type. Vocabulary list not found.")

        if os.path.isdir(args.in_text):
            text_files = glob(f"{args.in_text}/*.txt")
        else:
            text_files.append(args.in_text)
        pbar = tqdm(text_files, desc="Processing text files")
        for text in pbar:
            pbar.set_description(f"Currently processing: {text}")
            base_name = os.path.basename(text)[:-4]
            out_text_file = os.path.join(args.output_dir, base_name + ".txt")     
            split_text(
                text,
                out_text_file,
                pos_tagger=tagger,
                sat=sat,
                vocabulary=vocabulary,
                remove_asterisks=args.remove_asterisks,
                remove_brackets=args.remove_brackets,
                remove_curly_brackets=args.remove_curly_brackets,
                remove_parentheses=args.remove_parentheses,
                remove_number_in_brackets=args.remove_number_in_brackets,
                remove_number_in_curly_brackets=args.remove_number_in_curly_brackets,
                remove_number_in_parentheses=args.remove_number_in_parentheses,
                remove_number_in_guillumets=args.remove_number_in_guillumets,
                remove_speaker_labels=args.remove_speaker_labels,
                language=args.language,
                max_length=args.max_length,
                additional_split_symbols=args.additional_split_symbols,
                use_nemo_normalization=args.use_nemo_normalization,
                split_using_pattern=args.split_using_pattern,
                split_on_quotes=args.split_on_quotes,
                split_on_verbs=args.split_on_verbs,
                split_on_verbs_min_words=args.split_on_verbs_min_words,
                split_on_verbs_max_words=args.split_on_verbs_max_words,
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
            )
        print(f"Processed text saved at {args.output_dir}")

    if args.audio_dir:
        if not os.path.exists(args.audio_dir):
            raise ValueError(f"{args.audio_dir} not found. '--audio_dir' should contain .mp3 or .wav files.")

        audio_paths = glob(f"{args.audio_dir}/*")

        print(f"Processing {len(audio_paths)} audio files...")
        # Replace Parallel with a simple tqdm loop
        for audio_path in tqdm(audio_paths, desc="Processing audio files"):
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_wav_path = os.path.join(args.output_dir, base_name + ".wav")
            process_audio(
                in_file=audio_path,
                wav_file=output_wav_path,
                cut_prefix=args.cut_prefix,
                sample_rate=args.sample_rate,
                bit_depth=args.bit_depth,
            )

    print("Data preparation is complete.")
