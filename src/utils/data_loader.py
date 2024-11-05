import pandas as pd
import torch

    
def load_transcript_data(filepath):
    data = pd.read_csv(filepath)
    positions = data[['x', 'y']]
    cell_ids = data['CellId'].unique()
    labels = data['cancer']
    targets = data['target'] 

    return data, positions, cell_ids, labels, targets  



def load_cell_data(filepath):
    data = pd.read_csv(filepath)
    cell_positions = torch.tensor(data[['CenterX', 'CenterY']].values, dtype=torch.float)
    return data, cell_positions
