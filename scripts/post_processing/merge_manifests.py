import os
import json
import argparse
from collections import defaultdict


def process_merge_all_manifests(output_dir, model_names):
    """
    Merges all manifests into a single file.
    Args:
        output_dir: directory containing the manifests
        model_names: list of model names to merge manifests for
    """
    manifests = {model: f'manifest_{model}.json' for model in model_names}

    manifests_dir = os.path.join(output_dir, 'manifests')
    manifest_paths = {model: os.path.join(manifests_dir, fname) for model, fname in manifests.items()}

    common_keys = [
        'audio_filepath',
        'duration',
        'text',
        'text_no_preprocessing',
        'text_normalized',
        'score',
        'start_abs',
        'end_abs',
        'start',
        'end',
        'tokens',
    ]

    variable_keys = [
        'pred_text',
        'cer',
        'wer',
        'ins_rate',
        'del_rate',
        'sub_rate',
        'start_cer',
        'end_cer',
        'len_diff_ratio'
    ]

    # Store entries by model and audio_filepath
    entries_by_model = defaultdict(dict)

    for model, path in manifest_paths.items():
        if not os.path.exists(path):
            print(f"Manifest file not found, skipping: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, start=1):
                try:
                    entry = json.loads(line)
                    audio_fp = entry.get('audio_filepath')
                    if audio_fp:
                        entries_by_model[model][audio_fp] = entry
                    else:
                        print(f"[{model}] Missing 'audio_filepath' on line {lineno}, skipping.")
                except json.JSONDecodeError as e:
                    print(f"[{model}] Malformed JSON on line {lineno}, skipping. Error: {e}")

    # Get the union of all audio_filepaths across models
    all_audio_filepaths = set()
    for model_entries in entries_by_model.values():
        all_audio_filepaths.update(model_entries.keys())

    merged_manifest = []
    missing_models_log = []

    for audio_fp in sorted(all_audio_filepaths):
        merged_entry = {}
        available_models = [model for model in manifests if audio_fp in entries_by_model[model]]

        # Use common keys from the first available model
        if available_models:
            base_entry = entries_by_model[available_models[0]][audio_fp]
            for key in common_keys:
                merged_entry[key] = base_entry.get(key, None)

        # Append variable fields for each model (if the model has the entry)
        for model in manifests:
            model_entry = entries_by_model[model].get(audio_fp)
            if model_entry:
                for key in variable_keys:
                    merged_entry[f"{key}_{model}"] = model_entry.get(key, None)
            else:
                # Log or pad with None for missing model entry
                for key in variable_keys:
                    merged_entry[f"{key}_{model}"] = None
                missing_models_log.append((audio_fp, model))

        merged_manifest.append(merged_entry)

    # Save merged manifest
    merged_manifest_path = os.path.join(manifests_dir, 'merged_manifest.json')
    with open(merged_manifest_path, 'w', encoding='utf-8') as f:
        for entry in merged_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Merged manifest saved to {merged_manifest_path}")
    if missing_models_log:
        print(f"Warning: {len(missing_models_log)} entries were missing in one or more models.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge all manifests into a single file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory containing the manifests dir containing the individual manifest files.")
    parser.add_argument("--model_names", nargs='+', type=str, default=['nemo', 'wav2vec2_v3', 'faster_whisper_large'], help="List of model names to merge manifests for.")
    args = parser.parse_args()
    process_merge_all_manifests(args.output_dir, args.model_names)
