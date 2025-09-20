import pandas as pd
import re
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