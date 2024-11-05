import os
import pandas as pd

# Set the original directory for cell statistics and the new directory for processed files
original_dir = "/home/chaeeuncho_000724/liver_cancer/data/LiverCancerFiles/CellStatsDir"
new_dir = "../CancerLiverFiles/Cancer_Cell"

# Create new directory if it doesn't exist
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Traverse FOV folders to process CSV files
for folder_name in os.listdir(original_dir):
    folder_path = os.path.join(original_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith("FOV"):
        # Extract FOV number by removing leading zeros
        fov_number = folder_name[3:].lstrip('0')

        # Process each CSV file within the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Modify the 'CellId' column
                df['CellId'] = df['CellId'].apply(lambda x: f"C_{fov_number}_{x}")

                # Set the path to save in the new directory
                new_file_name = file_name.replace(f"F{folder_name[3:]}", f"C_{fov_number}")
                new_file_path = os.path.join(new_dir, new_file_name)

                # Save the modified CSV file
                df.to_csv(new_file_path, index=False)
                print(f"Processed and saved: {new_file_path}")

print("All files have been processed and saved.")
