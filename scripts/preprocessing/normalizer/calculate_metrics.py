import json
import jiwer

def calculate_metrics(manifest_path, edge_len=7):
    metadata = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())

                text = item.get("text", "")
                pred_text = item.get("pred_text", "")

                text = text.replace("\u200c", " ")
                pred_text = pred_text.replace("\u200c", " ")

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
                    start_text = text[:edge_len]
                    start_pred = pred_text[:edge_len]
                    end_text = text[-edge_len:]
                    end_pred = pred_text[-edge_len:]

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