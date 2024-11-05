import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, HypergraphConv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.data_loader import load_transcript_data, load_cell_data
from src.utils import create_knn_edges, create_hyperedge_knn_edges, save_metrics, save_model, plot_training_log, fov_contrastive_loss
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CompositeGraphNetWithFC(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=128, dropout=0.6):
        super(CompositeGraphNetWithFC, self).__init__()
        self.gcn_conv1 = GCNConv(in_channels, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.hyper_conv1 = HypergraphConv(in_channels, hidden_dim)
        self.hyper_conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, edge_weights, hyperedge_index, hyperedge_distances):
        x1 = F.relu(self.gcn_conv1(x, edge_index, edge_weight=edge_weights))
        x1 = self.dropout(x1)
        x1 = F.relu(self.gcn_conv2(x1, edge_index, edge_weight=edge_weights))

        # Aggregate hyperedge weights for each node
        node_hyperedge_weights = torch.zeros_like(x)
        for i, (node_idx, hyperedge_idx) in enumerate(zip(*hyperedge_index)):
            node_hyperedge_weights[node_idx] += hyperedge_distances[hyperedge_idx]

        x2 = x * node_hyperedge_weights
        x2 = F.relu(self.hyper_conv1(x2, hyperedge_index))
        x2 = self.dropout(x2)
        x2 = F.relu(self.hyper_conv2(x2, hyperedge_index))

        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

def train_model(model, train_loader, val_loader, optimizer, epochs=30, log_file='training_log.csv', accumulation_steps=2):
    scaler = GradScaler()
    model.train()
    train_loss_log = []
    val_loss_log = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                with autocast():
                    out = model(batch.x, batch.edge_index, batch.edge_weights, batch.hyperedge_index, batch.hyperedge_distances)
                    
                    # FOV embeddings
                    fov_embeddings = []
                    fov_labels = []
                    batch_size = batch.batch.max().item() + 1
                    for j in range(batch_size):
                        mask = (batch.batch == j)
                        fov_embedding = out[mask].mean(dim=0, keepdim=True)
                        fov_label = batch.y[j].view(-1)
                        fov_embeddings.append(fov_embedding)
                        fov_labels.append(fov_label)
                    
                    fov_embeddings = torch.cat(fov_embeddings, dim=0)
                    fov_labels = torch.cat(fov_labels, dim=0)
                    
                    # Calculate loss
                    cross_entropy_loss = F.cross_entropy(fov_embeddings, fov_labels)
                    cont_loss = fov_contrastive_loss(fov_embeddings, fov_labels)
                    
                    loss = cross_entropy_loss + cont_loss
                    loss = loss / accumulation_steps  # Normalize loss

                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
                pbar.update(1)

        train_loss_log.append(total_loss / len(train_loader))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Validation {epoch+1}/{epochs}', unit='batch') as pbar:
                for batch in val_loader:
                    batch = batch.to(device)
                    with autocast():
                        out = model(batch.x, batch.edge_index, batch.edge_weights, batch.hyperedge_index, batch.hyperedge_distances)
                        
                        # FOV embeddings
                        fov_embeddings = []
                        fov_labels = []
                        batch_size = batch.batch.max().item() + 1
                        for j in range(batch_size):
                            mask = (batch.batch == j)
                            fov_embedding = out[mask].mean(dim=0, keepdim=True)
                            fov_label = batch.y[j].view(-1)
                            fov_embeddings.append(fov_embedding)
                            fov_labels.append(fov_label)
                        
                        fov_embeddings = torch.cat(fov_embeddings, dim=0)
                        fov_labels = torch.cat(fov_labels, dim=0)
                        
                        # Calculate loss
                        cross_entropy_loss = F.cross_entropy(fov_embeddings, fov_labels)
                        cont_loss = fov_contrastive_loss(fov_embeddings, fov_labels)
                        
                        loss = cross_entropy_loss + cont_loss
                        val_loss += loss.item()
                        
                        pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})
                        pbar.update(1)
        
        val_loss_log.append(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}, Train Loss: {train_loss_log[-1]}, Val Loss: {val_loss_log[-1]}')
    
    pd.DataFrame({'train_loss': train_loss_log, 'val_loss': val_loss_log}).to_csv(log_file, index=False)

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            with autocast():
                out = model(batch.x, batch.edge_index, batch.edge_weights, batch.hyperedge_index, batch.hyperedge_distances)
                
                # FOV embeddings
                fov_embeddings = []
                fov_labels = []
                batch_size = batch.batch.max().item() + 1
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    fov_embedding = out[mask].mean(dim=0, keepdim=True)
                    fov_label = batch.y[i].view(-1)
                    fov_embeddings.append(fov_embedding)
                    fov_labels.append(fov_label)
                
                fov_embeddings = torch.cat(fov_embeddings, dim=0)
                fov_labels = torch.cat(fov_labels, dim=0)
                #pred = fov_embeddings.argmax(dim=1)
                
                # Softmax 적용
                probabilities = F.softmax(fov_embeddings, dim=1)
                # Argmax 적용
                pred = torch.argmax(probabilities, dim=1)
                
                y_true.extend(fov_labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f'F1 Score: {f1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

    metrics = {
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }
    save_metrics(metrics)

# Load data
transcript_data_path = '../data/updated_file.csv'
cell_data_path = '../data/Fov_1_Cell_location_2.csv'
transcript_data, positions, cell_ids, labels, targets = load_transcript_data(transcript_data_path)
cell_data, cell_positions = load_cell_data(cell_data_path)

# data 준비
scaler = MinMaxScaler()
scaled_transcript_data = transcript_data.copy()
scaled_transcript_data[['x', 'y']] = scaler.fit_transform(transcript_data[['x', 'y']])
data_list = []

for fov_id in transcript_data['fov'].unique():
    fov_data = transcript_data[transcript_data['fov'] == fov_id]
    scaled_fov_data = scaled_transcript_data[scaled_transcript_data['fov'] == fov_id]
    
    scaled_fov_data['target'] = scaled_fov_data['target'].astype(str)
    
    node_features = pd.get_dummies(scaled_fov_data[['x', 'y', 'target']], columns=['target']).astype(float)
    node_features = torch.tensor(node_features.values, dtype=torch.float32)
    
    labels = torch.tensor(fov_data['cancer'].values, dtype=torch.long)
    transcript_positions = fov_data[['x', 'y']].to_numpy()
    edge_index, edge_weights = create_knn_edges(transcript_positions, k=5)
    cell_positions = cell_data[cell_data['CellId'].isin(fov_data['CellId'])][['CenterX', 'CenterY']].values
    hyperedge_index, hyperedge_distances = create_hyperedge_knn_edges(cell_positions, k=5)

    # Debug print to check shapes
    print(f"hyperedge_index shape: {hyperedge_index.shape}")
    print(f"hyperedge_distances shape: {hyperedge_distances.shape}")

    incidence_matrix = torch.zeros((len(fov_data), len(cell_data['CellId'].unique())), dtype=torch.float32)
    for i, cell_id in enumerate(fov_data['CellId'].unique()):
        node_indices = fov_data[fov_data['CellId'] == cell_id].index.tolist()
        incidence_matrix[node_indices, i] = 1

    fov_label = torch.tensor([fov_data['cancer'].iloc[0]], dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index, edge_weights=edge_weights.float(), hyperedge_index=hyperedge_index, y=fov_label, hyperedge_distances=hyperedge_distances.float()).to(device)
    data_list.append(data)

# data가 하나뿐인 경우 처리
if len(data_list) == 1:
    train_data = val_data = test_data = data_list
    train_loader = val_loader = test_loader = GeoDataLoader(data_list, batch_size=1)
else:
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_loader = GeoDataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=2)
    test_loader = GeoDataLoader(test_data, batch_size=2)

model = CompositeGraphNetWithFC(in_channels=node_features.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

train_model(model, train_loader, val_loader, optimizer, epochs=30)
evaluate_model(model, test_loader)
save_model(model, 'final_model.pth')
plot_training_log('training_log.csv')
