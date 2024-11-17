import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Dataset class ===
class IoTDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

# === Preprocessing function ===
def preprocess_dataset(file_path):
    df = pd.read_csv(file_path)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    labeled_data, unlabeled_data, labeled_labels, _ = train_test_split(
        features, labels, test_size=0.8, stratify=labels, random_state=42
    )
    return labeled_data, labeled_labels, unlabeled_data

# === Generator and Discriminator classes ===
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

# === Loss functions ===
def supervised_loss(logits, labels, num_classes):
    criterion = nn.CrossEntropyLoss()
    return criterion(logits[:, :num_classes], labels)

def unsupervised_loss(logits, num_classes):
    real_loss = -torch.mean(torch.log(1 - torch.softmax(logits[:, num_classes], dim=0) + 1e-8))
    fake_loss = -torch.mean(torch.log(torch.softmax(logits[:, num_classes], dim=0) + 1e-8))
    return real_loss + fake_loss

def generator_loss(logits, num_classes):
    return -torch.mean(torch.log(1 - torch.softmax(logits[:, num_classes], dim=0) + 1e-8))

def reconstruction_loss(real_features, generated_features):
    return nn.MSELoss()(real_features, generated_features)

# === SS-GAN Model ===
class SSGAN:
    def __init__(self, noise_dim, data_dim, hidden_dim, num_classes, lr, lambda_recon):
        self.generator = Generator(noise_dim, hidden_dim, data_dim).cuda()
        self.discriminator = Discriminator(data_dim, hidden_dim, num_classes).cuda()

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.lambda_recon = lambda_recon

    def train_step(self, real_data, real_labels, unlabeled_data, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        real_data, real_labels = real_data.to(device), real_labels.to(device)
        unlabeled_data = unlabeled_data.to(device)
        
        # Train Discriminator
        real_logits, real_features = self.discriminator(real_data)
        loss_sup = supervised_loss(real_logits, real_labels, self.num_classes)

        unlabeled_logits, _ = self.discriminator(unlabeled_data)
        loss_unsup_real = unsupervised_loss(unlabeled_logits, self.num_classes)

        z = torch.randn(batch_size, self.generator.model[0].in_features).to(device)
        fake_data = self.generator(z)
        fake_logits, _ = self.discriminator(fake_data.detach())
        loss_unsup_fake = unsupervised_loss(fake_logits, self.num_classes)

        disc_loss = loss_sup + loss_unsup_real + loss_unsup_fake
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Train Generator
        fake_logits, fake_features = self.discriminator(fake_data)
        loss_g_unsup = generator_loss(fake_logits, self.num_classes)

        recon_loss = reconstruction_loss(real_features.detach(), fake_features)
        total_gen_loss = loss_g_unsup + self.lambda_recon * recon_loss
        
        self.gen_optimizer.zero_grad()
        total_gen_loss.backward()
        self.gen_optimizer.step()

        return disc_loss.item(), total_gen_loss.item()

# === Training Function ===
def train_ssgan(model, labeled_loader, unlabeled_loader, epochs, batch_size):
    for epoch in range(epochs):
        total_disc_loss, total_gen_loss = 0.0, 0.0
        for (labeled_data, labeled_labels), unlabeled_data in zip(labeled_loader, unlabeled_loader):
            disc_loss, gen_loss = model.train_step(labeled_data, labeled_labels, unlabeled_data, batch_size)
            total_disc_loss += disc_loss
            total_gen_loss += gen_loss
        print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {total_disc_loss:.4f}, Generator Loss: {total_gen_loss:.4f}")

# === Fitness Function for ABC ===
def fitness_function(params, pretrained_ssgan, test_loader):
    """
    Fitness function that evaluates the pretrained SS-GAN model with a given set of hyperparameters.
    The fitness value is defined as the difference between the predicted and actual values on the test dataset.

    :param params: Hyperparameter set being evaluated.
    :param pretrained_ssgan: Pretrained SS-GAN model.
    :param test_loader: DataLoader for test dataset.
    :return: Negative test loss (as fitness value).
    """
    # Update the model hyperparameters (e.g., dropout rate, noise_dim, etc.)
    pretrained_ssgan.lambda_recon = params["lambda_recon"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_ssgan.discriminator.eval()

    test_loss = 0.0

    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data, test_labels = test_data.to(device), test_labels.to(device)
            logits, _ = pretrained_ssgan.discriminator(test_data)
            loss = supervised_loss(logits, test_labels, pretrained_ssgan.num_classes)
            test_loss += loss.item()

    # Return negative test loss as fitness value
    return -test_loss


# === Main Function ===
if __name__ == "__main__":
    # Step 1: Define hyperparameter ranges
    param_ranges = {
        "batch_size": (8, 512),
        "learning_rate": (0.0001, 0.01),
        "epochs": (16, 512),
        "dropout_rate": (0, 0.5),
        "num_layers": (1, 10),
        "noise_dim": (1, 128),
        "lambda_recon": (0, 1),
    }

    # Step 2: Load dataset
    file_path = input("Enter dataset file path: ")
    labeled_data, labeled_labels, unlabeled_data = preprocess_dataset(file_path)
    
    # Split labeled data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        labeled_data, labeled_labels, test_size=0.2, stratify=labeled_labels, random_state=42
    )

    # Prepare DataLoader for training and testing
    train_dataset = IoTDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    test_dataset = IoTDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))
    unlabeled_dataset = IoTDataset(torch.tensor(unlabeled_data, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

    # Step 3: Train SS-GAN with manual hyperparameters
    print("Pretraining SS-GAN with manual hyperparameters...")
    manual_params = {
        "noise_dim": 64,
        "hidden_dim": 128,
        "lambda_recon": 0.1,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 64,
    }
    ssgan = SSGAN(
        noise_dim=manual_params["noise_dim"],
        data_dim=train_data.shape[1],
        hidden_dim=manual_params["hidden_dim"],
        num_classes=2,
        lr=manual_params["lr"],
        lambda_recon=manual_params["lambda_recon"]
    )
    train_ssgan(ssgan, train_loader, unlabeled_loader, epochs=manual_params["epochs"], batch_size=manual_params["batch_size"])

    # Step 4: Define fitness function for hyperparameter optimization
    def abc_fitness(params):
        return fitness_function(params, ssgan, test_loader)

    # Step 5: Optimize hyperparameters using Improved ABC
    print("Optimizing hyperparameters using Improved ABC...")
    abc = ImprovedABC(abc_fitness, param_ranges, num_bees=10, max_iter=5, limit=3)
    best_params = abc.optimize()
    print("Best Hyperparameters:", best_params)

    # Step 6: Train SS-GAN with optimized hyperparameters
    print("Training SS-GAN with optimized hyperparameters...")
    ssgan_optimized = SSGAN(
        noise_dim=int(best_params["noise_dim"]),
        data_dim=train_data.shape[1],
        hidden_dim=int(best_params["num_layers"]),
        num_classes=2,
        lr=best_params["learning_rate"],
        lambda_recon=best_params["lambda_recon"]
    )
    train_ssgan(ssgan_optimized, train_loader, unlabeled_loader, epochs=int(best_params["epochs"]), batch_size=int(best_params["batch_size"]))
