import os
import json
import argparse
import yaml
from tqdm import tqdm

def load_config(config_path):
    """Loads the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_models_from_manifest(manifest_data):
    """Infers model names from the keys in the first manifest entry."""
    if not manifest_data:
        return []
    
    first_entry = manifest_data[0]
    models = set()
    for key in first_entry.keys():
        if key.startswith('cer_'):
            models.add(key[len('cer_'):])
    return list(models)

def filter_samples(config, manifest_path, output_path):
    """
    Filters samples from a merged manifest based on criteria in the config.
    A sample is kept if it meets the criteria for at least one model.
    """
    # Load thresholds from config
    min_duration = config.get('min_duration', 0)
    cer_threshold = config.get('cer_threshold', 100)
    wer_threshold = config.get('wer_threshold', 100)
    cer_edge_threshold = config.get('cer_edge_threshold', 100)
    len_diff_ratio_threshold = config.get('len_diff_ratio_threshold', 1.0)

    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = [json.loads(line) for line in f]

    models = get_models_from_manifest(manifest_data)
    if not models:
        print("No models found in the manifest. Exiting.")
        return

    print(f"Found models: {', '.join(models)}")
    
    filtered_manifest = []
    for entry in tqdm(manifest_data, desc="Filtering samples"):
        # Global check for duration
        if entry.get('duration', 0) <= min_duration:
            continue

        # Check if the sample passes the criteria for at least one model
        passed_for_one_model = False
        for model in models:
            # Check if all required keys for the model are present
            required_keys = [
                f'cer_{model}', f'wer_{model}', f'start_cer_{model}',
                f'end_cer_{model}', f'len_diff_ratio_{model}'
            ]
            if any(entry.get(key) is None for key in required_keys):
                continue

            # Check criteria for the current model
            if (entry[f'cer_{model}'] <= cer_threshold and
                entry[f'wer_{model}'] <= wer_threshold and
                entry[f'start_cer_{model}'] <= cer_edge_threshold and
                entry[f'end_cer_{model}'] <= cer_edge_threshold and
                entry[f'len_diff_ratio_{model}'] <= len_diff_ratio_threshold):
                
                passed_for_one_model = True
                break  # No need to check other models for this entry

        if passed_for_one_model:
            filtered_manifest.append(entry)

    # Save filtered manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in filtered_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Filtering complete. Kept {len(filtered_manifest)} out of {len(manifest_data)} samples.")
    print(f"Filtered manifest saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter samples from a merged manifest.")
    parser.add_argument("--config_path", required=True, type=str, help="Path to the config.yaml file.")
    parser.add_argument("--manifest_path", required=True, type=str, help="Path to the merged_manifest.json file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the filtered manifest. Defaults to the directory of the input manifest.")
    
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.manifest_path)
        
    output_path = os.path.join(output_dir, 'filtered_manifest.json')

    config = load_config(args.config_path)
    filter_samples(config, args.manifest_path, output_path)
