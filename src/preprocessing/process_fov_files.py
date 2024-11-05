import os
import pandas as pd

# Set the original directory with raw data and a new directory for processed files
original_dir = "/home/chaeeuncho_000724/liver_cancer/data/LiverCancerFiles/AnalysisResults/0a0a0a0a0a"
new_dir = "../CancerLiverFiles/Cancer_Fov"

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
            if file_name.endswith("complete_code_cell_target_call_coord.csv"):
                file_path = os.path.join(folder_path, file_name)
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Modify the 'fov' column
                df['fov'] = df['fov'].apply(lambda x: f"C_{fov_number}")

                # Modify the 'CellId' column
                df['CellId'] = df['CellId'].apply(lambda x: f"C_{fov_number}_{x}")

                # Save the modified CSV file
                new_file_name = file_name.replace(f"FOV{folder_name[3:]}", f"C_{fov_number}")
                new_file_path = os.path.join(new_dir, new_file_name)

                # Save the file
                df.to_csv(new_file_path, index=False)
                print(f"Processed and saved: {new_file_path}")

print("All files have been processed and saved.")
