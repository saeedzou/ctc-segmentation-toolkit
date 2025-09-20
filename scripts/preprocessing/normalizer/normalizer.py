import pandas as pd
import re
import json
import sys
import argparse
import jiwer
from parsnorm import ParsNorm


# Normalizations
def map_words_dict(csv_file="map_words.csv"):
    map_df = pd.DataFrame(pd.read_csv(csv_file))
    mapping_dict = {}
    for i in range(len(map_df['original'])):
        mapping_dict[map_df['original'][i]] = map_df['corrected'][i]
    return {str(k): str(v) for k, v in mapping_dict.items() if pd.notna(k) and pd.notna(v)}

def create_pattern_from_mapping_dict(mapping_dict):
    return "|".join(map(re.escape, mapping_dict.keys()))

def multiple_replace(text, mapping_dict, mapping_pattern):
    return re.sub(mapping_pattern, lambda m: mapping_dict[m.group()], str(text))


class Normalizer:
    def __init__(self, edge_len=7, map_words_path='scripts/preprocessing/normalizer/final_map_words.csv'):
        self.chars_mapping_dict = {"\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ", "\u202b": " "}
        self.words_mapping_dict = map_words_dict(map_words_path)
        self.chars_mapping_pattern = create_pattern_from_mapping_dict(self.chars_mapping_dict)
        self.words_mapping_pattern = create_pattern_from_mapping_dict(self.words_mapping_dict)

        self.norm_obj = ParsNorm()
        self.edge_len = edge_len
    
    def normalize(self, text):
        text = self.norm_obj.normalize(text)
        text = multiple_replace(text, self.chars_mapping_dict, self.chars_mapping_pattern)
        text = multiple_replace(text, self.words_mapping_dict, self.words_mapping_pattern)
        return text
        
    def normalize_manifest(self, manifest_path):
        metadata = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())

                    text = item.get("text_no_preprocessing", "")
                    pred_text = item.get("pred_text", "")

                    text = self.normalize(text)
                    pred_text = self.normalize(pred_text)

                    # Compute detailed WER info
                    if text.strip() == "":
                        cer = len(pred_text) if pred_text.strip() else 0
                        wer = len(pred_text.split()) if pred_text.strip() else 0
                        del_rate = 0.0
                        ins_rate = float('inf')
                        sub_rate = 0.0
                        start_cer = 0.0
                        end_cer = 0.0
                        len_diff_ratio = 1.0 * abs(len(pred_text) - len(text)) / max(len(text), 1e-9)
                    else:
                        cer = jiwer.cer(text, pred_text)
                        measures = jiwer.compute_measures(text, pred_text)
                        errors = (
                            measures['insertions'] +
                            measures['deletions'] +
                            measures['substitutions']
                        )
                        total_words = len(text.split())

                        wer = errors / total_words
                        ins_rate = measures['insertions'] / total_words
                        del_rate = measures['deletions'] / total_words
                        sub_rate = measures['substitutions'] / total_words
                        # calculate the cer of the first edge_len characters
                        start_text = text[:self.edge_len]
                        start_pred = pred_text[:self.edge_len]
                        end_text = text[-self.edge_len:]
                        end_pred = pred_text[-self.edge_len:]

                        start_cer = jiwer.cer(start_text, start_pred) if start_text.strip() else 1.0
                        end_cer = jiwer.cer(end_text, end_pred) if end_text.strip() else 1.0

                        len_diff_ratio = 1.0 * abs(len(pred_text) - len(text)) / max(len(text), 1e-9)
                

                    item['text'] = text
                    item['pred_text_normalized'] = pred_text
                    item['cer'] = cer
                    item['wer'] = wer
                    item['ins_rate'] = ins_rate
                    item['del_rate'] = del_rate
                    item['sub_rate'] = sub_rate
                    item['start_cer'] = start_cer
                    item['end_cer'] = end_cer
                    item['len_diff_ratio'] = len_diff_ratio

                    metadata.append(item)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} in line: {line}")

        # Write the updated metadata back to the same file
        with open(manifest_path, "w", encoding="utf-8") as f:
            for item in metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")