import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from dinov2.models.vision_transformer import vit_large
from dinov2.hub.backbones import dinov2_vitg14, dinov2_vitg14_reg
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import seaborn as sns

def compute_average_cosine_similarities(indices, patch_embeddings,  num_patches):
    # Neighbor cosine similarity
    grid_size = int(num_patches**0.5)
    average_similarities = []
    for idx in zip(*indices):
        i, j = idx[1], idx[2]
        token = patch_embeddings[0, i, j]
        neighbors = []

        if i > 0: neighbors.append(patch_embeddings[0, i-1, j])
        if i < grid_size - 1: neighbors.append(patch_embeddings[0, i+1, j])
        if j > 0: neighbors.append(patch_embeddings[0, i, j-1])
        if j < grid_size - 1: neighbors.append(patch_embeddings[0, i, j+1])
        if neighbors:
            cosine_sims = [
                F.cosine_similarity(token.unsqueeze(0), neighbor.unsqueeze(0)).item()
                for neighbor in neighbors
            ]
            average_similarities.append(sum(cosine_sims) / len(cosine_sims))
    return average_similarities



def run_3glo_l(seed):
    # logistic regression norm-normal 3 seeds dinov2 large
    train_subset_size = 800
    test_subset_size = 200
    batch_size = 100
    print(f"Running experiment with seed {seed}...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = pth_transforms.Compose([
        pth_transforms.Resize((196, 196)),
        pth_transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:train_subset_size]
    test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print("Subsets and DataLoaders ready.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_large(patch_size=14, img_size=526, init_values=1.0, block_chunks=0)
    model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))
    model.to(device)
    model.eval()

    print("DINOv2 model loaded and ready.")

    high_norm_tokens = []
    normal_tokens = []
    labels_high = []
    labels_normal = []

    print("Extracting patch embeddings...")
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)

        with torch.no_grad():
            features = model.forward_features(images)["x_prenorm"]
            patch_features = features[:, 1:, :]
            norms = torch.norm(patch_features, dim=-1).cpu().numpy()

            for i in range(images.size(0)):
                single_patch_features = patch_features[i]
                single_norms = norms[i]
                high_indices = np.where(single_norms > 100)[0]
                normal_indices = np.where(single_norms <= 100)[0]

                if high_indices.size > 0:
                    high_norm_token = single_patch_features[high_indices[np.random.randint(len(high_indices))], :].cpu().numpy()
                    high_norm_tokens.append(high_norm_token)
                    labels_high.append(targets[i].item())

                if normal_indices.size > 0:
                    normal_token = single_patch_features[normal_indices[np.random.randint(len(normal_indices))], :].cpu().numpy()
                    normal_tokens.append(normal_token)
                    labels_normal.append(targets[i].item())

        print(f"Processed batch {batch_idx + 1}/{len(train_loader)}")

    print("Patch embedding extraction complete.")

    X_high = np.array(high_norm_tokens)
    X_normal = np.array(normal_tokens)
    y_high = np.array(labels_high)
    y_normal = np.array(labels_normal)

    print(len(X_high), len(X_normal))
    print(len(y_high), len(y_normal))

    X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(X_high, y_high, test_size=0.2, random_state=seed)
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=seed)

    print("Training Logistic Regression classifiers...")
    clf_high = LogisticRegression(max_iter=1000).fit(X_high_train, y_high_train)
    clf_normal = LogisticRegression(max_iter=1000).fit(X_normal_train, y_normal_train)

    print("Evaluating classifiers...")
    high_acc = accuracy_score(y_high_test, clf_high.predict(X_high_test))
    normal_acc = accuracy_score(y_normal_test, clf_normal.predict(X_normal_test))

    print(f"High-Norm Token Accuracy: {high_acc:.4f}")
    print(f"Normal Token Accuracy: {normal_acc:.4f}")

    return high_acc, normal_acc

def run_3loc(seed, X_high, X_normal, y_high, y_normal):

    # logistic regression for positional information 3 seeds
    X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(
        X_high, y_high, test_size=0.2, random_state=seed
    )
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(
        X_normal, y_normal, test_size=0.2, random_state=seed
    )

    high_model = LinearRegression().fit(X_high_train, y_high_train)
    normal_model = LinearRegression().fit(X_normal_train, y_normal_train)

    high_pred = high_model.predict(X_high_test)
    high_avg_distance = np.mean(np.linalg.norm(high_pred - y_high_test, axis=1))
    high_top1_acc = top1_accuracy(high_pred, y_high_test)

    normal_pred = normal_model.predict(X_normal_test)
    normal_avg_distance = np.mean(np.linalg.norm(normal_pred - y_normal_test, axis=1))
    normal_top1_acc = top1_accuracy(normal_pred, y_normal_test)

    return high_top1_acc, high_avg_distance, normal_top1_acc, normal_avg_distance

def extract_features(model, data_loader):
    # extract features for performance evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.numpy()

            outputs = model.forward_features(images)["x_prenorm"][:, 0, :]
            features.append(outputs.cpu().numpy())
            labels.append(targets)
    return np.vstack(features), np.concatenate(labels)

def evaluate_linear_probe(train_features, train_labels, test_features, test_labels):
    # evaluate linear probe classifier
    clf = LogisticRegression(max_iter=1000).fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


def run_3glo_g(seed):
    # logistic regression norm-normal 3 seeds dinov2 giant
    train_subset_size = 800
    test_subset_size = 200
    batch_size = 100
    print(f"Running experiment with seed {seed}...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = pth_transforms.Compose([
        pth_transforms.Resize((196, 196)),
        pth_transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(train_dataset))[:train_subset_size]
    test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print("Subsets and DataLoaders ready.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14(patch_size=14, img_size=526, init_values=1.0, block_chunks=0)
    model.load_state_dict(torch.load('dinov2_vitg14_pretrain.pth'))
    model.to(device)
    model.eval()

    print("DINOv2 model loaded and ready.")

    high_norm_tokens = []
    normal_tokens = []
    labels_high = []
    labels_normal = []

    print("Extracting patch embeddings...")
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)

        with torch.no_grad():
            features = model.forward_features(images)["x_prenorm"]
            patch_features = features[:, 1:, :]
            norms = torch.norm(patch_features, dim=-1).cpu().numpy()

            for i in range(images.size(0)):
                single_patch_features = patch_features[i]
                single_norms = norms[i]
                high_indices = np.where(single_norms > 100)[0]
                normal_indices = np.where(single_norms <= 100)[0]

                if high_indices.size > 0:
                    high_norm_token = single_patch_features[high_indices[np.random.randint(len(high_indices))], :].cpu().numpy()
                    high_norm_tokens.append(high_norm_token)
                    labels_high.append(targets[i].item())

                if normal_indices.size > 0:
                    normal_token = single_patch_features[normal_indices[np.random.randint(len(normal_indices))], :].cpu().numpy()
                    normal_tokens.append(normal_token)
                    labels_normal.append(targets[i].item())

        print(f"Processed batch {batch_idx + 1}/{len(train_loader)}")

    print("Patch embedding extraction complete.")

    X_high = np.array(high_norm_tokens)
    X_normal = np.array(normal_tokens)
    y_high = np.array(labels_high)
    y_normal = np.array(labels_normal)

    print(len(X_high), len(X_normal))
    print(len(y_high), len(y_normal))

    X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(X_high, y_high, test_size=0.2, random_state=seed)
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=seed)

    print("Training Logistic Regression classifiers...")
    clf_high = LogisticRegression(max_iter=1000).fit(X_high_train, y_high_train)
    clf_normal = LogisticRegression(max_iter=1000).fit(X_normal_train, y_normal_train)

    print("Evaluating classifiers...")
    high_acc = accuracy_score(y_high_test, clf_high.predict(X_high_test))
    normal_acc = accuracy_score(y_normal_test, clf_normal.predict(X_normal_test))

    print(f"High-Norm Token Accuracy: {high_acc:.4f}")
    print(f"Normal Token Accuracy: {normal_acc:.4f}")

    return high_acc, normal_acc

def top1_accuracy(pred_positions, true_positions):
    # calculate top-1 accuracy
    correct = np.all(np.round(pred_positions) == true_positions, axis=1)
    return np.mean(correct) * 100


def extract_features_loc(model, data_loader):
    # extract positional information
    patch_size = 14
    positions = []
    embeddings = []
    norms = []

    print("Extracting patch embeddings and positional information...")
    for images, _ in data_loader:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        images = images.to(device)
        with torch.no_grad():
            features = model.forward_features(images)["x_prenorm"][:, 1:, :]
            patch_norms = torch.norm(features, dim=-1).cpu().numpy()
            img_h, img_w = images.shape[-2:]
            patch_h, patch_w = img_h // patch_size, img_w // patch_size

            for i in range(features.size(0)):
                patches = features[i].cpu().numpy()
                for idx, patch in enumerate(patches):
                    row, col = divmod(idx, patch_w)
                    positions.append((row, col))
                    embeddings.append(patch)
                    norms.append(patch_norms[i, idx])
    print("Complete.")
    return np.array(embeddings), np.array(positions), np.array(norms)