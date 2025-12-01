import pandas as pd
import os

# Folder containing the original CSVs
input_folder = "moral_stories_csv"

# Folder to save the cleaned CSVs
output_folder = "moral_stories_cleaned"
os.makedirs(output_folder, exist_ok=True)

# Process each CSV in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        path = os.path.join(input_folder, filename)
        df = pd.read_csv(path)

        # Skip empty files
        if df.empty:
            continue

        # Extract the common story ID prefix (all but last character)
        df['story_prefix'] = df['ID'].astype(str).str[:-1]

        # Merge moral and immoral actions
        merged = df.groupby('story_prefix').agg({
            'moral_action': 'first',
            'immoral_action': 'first'
        }).reset_index()

        merged.rename(columns={'story_prefix': 'story_id'}, inplace=True)

        # Save cleaned CSV
        cleaned_path = os.path.join(output_folder, filename)
        merged.to_csv(cleaned_path, index=False)
        print(f"Cleaned {filename} -> {cleaned_path}")

print("All CSVs cleaned and saved.")
