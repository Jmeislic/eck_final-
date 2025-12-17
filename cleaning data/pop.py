# To make this function I asked ChatGPT: make a python script that takes in a csv filer called data_With_sentence.csv and deletes every row with any of these norm:,situation:,intention,moral_action:,moral_consequence:,label:,immoral_action:,immoral_consequence:, so an or statement for any of those

import pandas as pd

def clean_csv_data(input_filename="data_With_sentence.csv", output_filename="cleaned_data.csv"):
    """
    Reads a CSV, deletes rows containing specified text in ANY of the given columns,
    and saves the result to a new file.
    """
    try:
        # 1. Load the CSV file
        df = pd.read_csv(input_filename)
        print(f"Original number of rows: {len(df)}")

        # 2. Define the list of strings to look for
        # Use placeholders (col_a_filter, col_b_filter) instead of sensitive terms
        strings_to_delete = ['norm:','situation:','intention:','moral_action:','moral_consequence:','label:','immoral_action:','immoral_consequence:']

        # 3. Create a boolean mask for rows to KEEP (i.e., rows that do NOT contain any of the strings)
        # The | (or) operator combines the checks for different strings across all columns
        mask_to_keep = pd.Series([True] * len(df))

        for col in df.columns:
            # For each string, check if the string is contained in the current column
            for s in strings_to_delete:
                # Use str.contains() with case=False and na=False to handle missing values and ignore case
                # The ~ (tilde) negates the condition, so we mark rows that *do not* contain the string
                mask_to_keep &= ~df[col].astype(str).str.contains(s, case=False, na=False)

        # 4. Apply the mask to filter the DataFrame
        df_cleaned = df[mask_to_keep]
        print(f"Number of rows after cleaning: {len(df_cleaned)}")

        # 5. Save the cleaned DataFrame to a new CSV file
        df_cleaned.to_csv(output_filename, index=False)
        print(f"Cleaned data saved to {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage (you would need to create a dummy CSV or change the filename)
clean_csv_data()