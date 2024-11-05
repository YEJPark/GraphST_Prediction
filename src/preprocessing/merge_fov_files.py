import os
import pandas as pd

# Set the directory path for merging files
input_dir = "../data/fov/"
output_file_csv = "../data/final/fov_final.csv"

# Read all CSV files into a list of dataframes
dataframes = []
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Combine all dataframes into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a CSV file
merged_df.to_csv(output_file_csv, index=False)
print(f"Merged file saved as CSV: {output_file_csv}")

# Print input and result examples
print("Input example:")
for df in dataframes[:2]:  # Print examples of the first two dataframes
    print(df.head(), "\n")

print("Merged result example:")
print(merged_df.head())
