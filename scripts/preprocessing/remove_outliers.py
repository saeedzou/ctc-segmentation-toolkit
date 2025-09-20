import os
import jiwer
import argparse
import re
from scripts.preprocessing.normalizer.normalizer import Normalizer

def main(args):
    min_wer = args.min_wer / 100
    min_cer = args.min_cer / 100
    processed_dir = os.path.join(args.output_dir, 'processed')

    text_dir = os.path.join(processed_dir, 'text')
    hypothesis_dir = os.path.join(processed_dir, 'hypothesis')
    audio_dir = os.path.join(processed_dir, 'audio')

    if not os.path.isdir(hypothesis_dir):
        print(f"Hypothesis directory not found at {hypothesis_dir}, skipping outlier removal.")
        return

    # Get all hypothesis base names
    hypothesis_basenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(hypothesis_dir)
        if f.endswith('.txt')
    }

    # Get all text base names from triplets (ignore the _with_punct stuff)
    text_basenames = {
        f.replace('_with_punct_normalized', '')
         .replace('_with_punct', '')
         .replace('.txt', '')
        for f in os.listdir(text_dir)
        if f.endswith('.txt')
    }

    # --- Remove text triplets with no corresponding hypothesis ---
    for base_name in text_basenames:
        if base_name not in hypothesis_basenames:
            for suffix in ['', '_with_punct', '_with_punct_normalized']:
                to_remove = os.path.join(text_dir, f"{base_name}{suffix}.txt")
                if os.path.exists(to_remove):
                    os.remove(to_remove)
                    print(f"Removed text file: {to_remove}")

    # --- Remove hypothesis files with no corresponding base text file ---
    for base_name in hypothesis_basenames:
        if base_name not in text_basenames:
            to_remove = os.path.join(hypothesis_dir, f"{base_name}.txt")
            if os.path.exists(to_remove):
                os.remove(to_remove)
                print(f"Removed hypothesis file: {to_remove}")
            to_remove = os.path.join(audio_dir, f"{base_name}.wav")
            if os.path.exists(to_remove):
                os.remove(to_remove)
                print(f"Removed audio file: {to_remove}")

    normalizer = Normalizer()

    for hypo_file in os.listdir(hypothesis_dir):
        if hypo_file.endswith('.txt'):
            base_name = os.path.splitext(hypo_file)[0]
            hypo_path = os.path.join(hypothesis_dir, hypo_file)
            ref_path = os.path.join(text_dir, f"{base_name}_with_punct.txt")

            if os.path.exists(ref_path):
                with open(ref_path, 'r', encoding='utf-8') as ref_f, open(hypo_path, 'r', encoding='utf-8') as hypo_f:
                    ref_text = ref_f.read().strip()
                    ref_text = normalizer.normalize(re.sub(r"\s+", " ", ref_text))
                    hypo_text = hypo_f.read().strip()
                    hypo_text = normalizer.normalize(re.sub(r"\s+", " ", hypo_text))

                    if not ref_text or not hypo_text:
                        # remove hypothesis
                        os.remove(hypo_path)
                        print(f"Removed empty/hypothesis file: {hypo_path}")
                        # remove text triplets
                        for suffix in ['', '_with_punct', '_with_punct_normalized']:
                            to_remove = os.path.join(text_dir, f"{base_name}{suffix}.txt")
                            if os.path.exists(to_remove):
                                os.remove(to_remove)
                                print(f"Removed text file: {to_remove}")
                        # remove audio
                        audio_path = os.path.join(audio_dir, f"{base_name}.wav")
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            print(f"Removed audio file: {audio_path}")
                    else:
                        wer = jiwer.wer(ref_text, hypo_text)
                        cer = jiwer.cer(ref_text, hypo_text)
                        print(f"[{hypo_file}] WER: {wer:.3f}, CER: {cer:.3f}")

                        if not((wer < min_wer) and (cer < min_cer)):
                            # remove hypothesis
                            os.remove(hypo_path)
                            print(f"Removed high-WER hypothesis file: {hypo_path}")
                            # remove text triplets
                            for suffix in ['', '_with_punct', '_with_punct_normalized']:
                                to_remove = os.path.join(text_dir, f"{base_name}{suffix}.txt")
                                if os.path.exists(to_remove):
                                    os.remove(to_remove)
                                    print(f"Removed text file: {to_remove}")
                            # remove audio
                            audio_path = os.path.join(audio_dir, f"{base_name}.wav")
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                                print(f"Removed audio file: {audio_path}")
    print("Completed removing outliers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove outliers from processed data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the 'processed' folder.")
    parser.add_argument("--min_wer", type=float, default=80, help="Minimum WER threshold for removing hypotheses.")
    parser.add_argument("--min_cer", type=float, default=50, help="Minimum CER threshold for removing hypotheses.")
    args = parser.parse_args()
    main(args)
