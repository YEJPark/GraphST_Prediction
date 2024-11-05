import pandas as pd

# Load the CSV file in chunks and add a 'cancer' column with all values set to 1
file_path = '../data/C_Fov_merged_file.csv'
chunks = pd.read_csv(file_path, chunksize=10000)

# Add 'cancer' column to each chunk and store in a list
chunk_list = []
for chunk in chunks:
    chunk['cancer'] = 1
    chunk_list.append(chunk)

# Combine the chunks back into a single dataframe
df = pd.concat(chunk_list)

# Save the modified dataframe to a new CSV file
output_file_path = '../data/fov/C_Fov_merged_file_with_cancer.csv'
df.to_csv(output_file_path, index=False)

print(df.head())
