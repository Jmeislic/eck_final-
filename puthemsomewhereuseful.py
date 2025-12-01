import os
import json
import pandas as pd
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

data_dir = snapshot_download(
    "demelin/moral_stories",
    repo_type="dataset",
    allow_patterns=["**/*.jsonl", "**/*.json", "*.json", "*.jsonl"]
)

print("Downloaded to:", data_dir)
import os
import json
import pandas as pd

output_dir = "./moral_stories_csv"
os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            path = os.path.join(root, filename)

            # Load data
            if filename.endswith(".jsonl"):
                rows = [json.loads(line) for line in open(path)]
                df = pd.DataFrame(rows)
            else:
                data = json.load(open(path))
                if isinstance(data, dict):
                    data = data.get("data", data)
                df = pd.DataFrame(data)

            # Save CSV
            rel_path = os.path.relpath(path, data_dir)
            csv_name = rel_path.replace("/", "_").replace(".jsonl", ".csv").replace(".json", ".csv")
            csv_path = os.path.join(output_dir, csv_name)

            df.to_csv(csv_path, index=False)
            print("Saved:", csv_path)
