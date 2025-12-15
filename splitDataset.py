#To make this file I asked chatGPT "Can you make me a python file which takes in the dataset "data_With_Sentence.csv", then shuffles it, and splits it into two datasets csv's with an 80-20 split, one named "final_train", the second named "final_test"."
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load the dataset
    input_file = "extra_cleaned_data.csv"
    df = pd.read_csv(input_file)

    # Shuffle and split the dataset (80% train, 20% test)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # Save the split datasets
    train_df.to_csv("final_train.csv", index=False)
    test_df.to_csv("final_test.csv", index=False)

    print("Dataset successfully split!")
    print(f"Training set: {len(train_df)} rows -> final_train.csv")
    print(f"Test set: {len(test_df)} rows -> final_test.csv")

if __name__ == "__main__":
    main()
