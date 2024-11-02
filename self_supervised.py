import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(data, epochs=100, batch_size=32, hidden_dim=64):
    # Convert data to PyTorch tensor
    data_tensor = torch.FloatTensor(data)
    input_dim = data.shape[1]
    
    # Create dataloader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize autoencoder
    autoencoder = Autoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            
            # Forward pass
            encoded, decoded = autoencoder(inputs)
            loss = criterion(decoded, inputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return autoencoder

def get_pseudo_labels(data, autoencoder, n_clusters):
    # Get encoded features
    data_tensor = torch.FloatTensor(data)
    encoded_features, _ = autoencoder(data_tensor)
    encoded_features = encoded_features.detach().numpy()
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pseudo_labels = kmeans.fit_predict(encoded_features)
    
    return pseudo_labels, encoded_features