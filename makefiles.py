import json
import pandas as pd
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# download full dataset
data_dir = snapshot_download(
    "demelin/moral_stories",
    repo_type="dataset"
)

def convert_json_to_csv(path):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)

    elif path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # flatten if needed
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        df = pd.DataFrame(data)

    csv_path = path.replace(".jsonl", ".csv").replace(".json", ".csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)


# recursively convert every JSON/JSONL file
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith(".json") or f.endswith(".jsonl"):
            full_path = os.path.join(root, f)
            print("Converting:", full_path)
            convert_json_to_csv(full_path)
