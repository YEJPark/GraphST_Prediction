import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

def create_knn_edges(positions, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    edge_index = []
    edge_weights = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            edge_index.append([i, indices[i, j]])
            edge_index.append([indices[i, j], i])
            edge_weights.append(distances[i, j])
            edge_weights.append(distances[i, j])
    return torch.tensor(edge_index).t(), torch.tensor(edge_weights)

def create_hyperedge_knn_edges(cell_positions, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(cell_positions)
    distances, indices = nbrs.kneighbors(cell_positions)
    edge_index = []
    edge_weights = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            edge_index.append([i, indices[i, j]])
            edge_index.append([indices[i, j], i])
            edge_weights.append(distances[i, j])
            edge_weights.append(distances[i, j])
    return torch.tensor(edge_index).t(), torch.tensor(edge_weights)

def save_metrics(metrics, filename='metrics.txt'):
    with open(filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def plot_training_log(log_file):
    data = pd.read_csv(log_file)
    plt.figure(figsize=(10, 5))
    plt.plot(data['train_loss'], label='Train Loss')
    plt.plot(data['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_log_plot.png')
    plt.show()

def fov_contrastive_loss(embeddings, labels, margin=1.0):
    """
    Contrastive loss 계산 함수.
    Args:
        embeddings: model의 출력 임베딩
        labels: 노드 레이블
        margin: 마진 값
    Returns:
        Contrastive loss 값
    """
    n = embeddings.size(0)
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    positive_pairs = distance_matrix[labels_matrix]
    negative_pairs = distance_matrix[~labels_matrix]
    
    positive_loss = torch.mean(positive_pairs)
    negative_loss = torch.mean(F.relu(margin - negative_pairs))
    
    loss = positive_loss + negative_loss
    return loss
