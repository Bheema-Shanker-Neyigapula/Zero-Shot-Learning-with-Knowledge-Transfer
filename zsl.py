import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gensim.downloader as api
import matplotlib.pyplot as plt

# Step 1: Data Collection and Preprocessing from Synthetic Data

def generate_synthetic_data(num_classes=10, num_samples_per_class=1000, feature_dim=300):
    num_samples = num_classes * num_samples_per_class
    data = np.random.randn(num_samples, feature_dim)
    labels = np.repeat(np.arange(num_classes), num_samples_per_class)
    return data, labels

# Step 2: Semantic Representation Construction using Word2Vec model

def construct_semantic_representations(class_names, word2vec_model):
    # Construct semantic representations for each class
    semantic_representations = []
    for class_name in class_names:
        words = class_name.split('_')  # Assuming class names have multiple words separated by underscores
        class_representation = np.mean([word2vec_model[word] for word in words if word in word2vec_model], axis=0)
        semantic_representations.append(class_representation)

    return np.array(semantic_representations)

# Step 3: Zero-Shot Learning Models and Architectures

class ZSLModel(nn.Module):
    def __init__(self, num_classes, semantic_embeddings):
        super(ZSLModel, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(300, num_classes)
        self.semantic_embeddings = torch.FloatTensor(semantic_embeddings)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        scores = torch.mm(x, self.semantic_embeddings.t())
        return scores

# Step 4: Knowledge Transfer Techniques and Implementation

def transfer_knowledge(zsl_model, new_model):
    # Transfer knowledge from the ZSL model to the new model
    new_model.fc.weight.data = zsl_model.fc.weight.data.clone()
    new_model.fc.bias.data = zsl_model.fc.bias.data.clone()

# Step 5: Evaluation Setup and Metrics

def evaluate_model(model, data, labels):
    model.eval()
    with torch.no_grad():
        data = torch.tensor(data).float()
        labels = torch.tensor(labels).long()
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(labels, predicted)
        return accuracy

if __name__ == "__main__":
    num_classes_list = [5, 10, 15, 20, 25]  # Vary the number of classes
    num_samples_per_class = 1000
    feature_dim = 300
    word2vec_model = api.load("word2vec-google-news-300")

    zsl_model_accuracies = []
    new_model_accuracies = []

    for num_classes in num_classes_list:
        # Step 1: Data Collection and Preprocessing from Synthetic Data
        data, labels = generate_synthetic_data(num_classes, num_samples_per_class, feature_dim)

        # Step 2: Semantic Representation Construction using Word2Vec model
        class_names = [f"class_{i}" for i in range(num_classes)]
        semantic_embeddings = construct_semantic_representations(class_names, word2vec_model)

        # Step 3: Zero-Shot Learning Models and Architectures
        zsl_model = ZSLModel(num_classes, semantic_embeddings)

        # Step 5: Evaluation Setup and Metrics for ZSL Model
        validation_data, validation_labels = generate_synthetic_data(num_classes, num_samples_per_class, feature_dim)
        zsl_accuracy = evaluate_model(zsl_model, validation_data, validation_labels)

        zsl_model_accuracies.append(zsl_accuracy)
        print(f"Number of classes: {num_classes}, ZSL Accuracy: {zsl_accuracy:.2f}")

        # Step 4: Knowledge Transfer Techniques and Implementation
        new_model = ZSLModel(num_classes, semantic_embeddings)  # Create a new model for knowledge transfer
        transfer_knowledge(zsl_model, new_model)

        # Step 5: Evaluation Setup and Metrics for New Model with Knowledge Transfer
        new_model_accuracy = evaluate_model(new_model, validation_data, validation_labels)

        new_model_accuracies.append(new_model_accuracy)
        print(f"Number of classes: {num_classes}, New Model Accuracy with Knowledge Transfer: {new_model_accuracy:.2f}")

    # Plot the results
    plt.plot(num_classes_list, zsl_model_accuracies, label="ZSL Model", marker='o', color='blue')
    plt.plot(num_classes_list, new_model_accuracies, label="New Model with Knowledge Transfer", marker='x', color='orange')
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy")
    plt.title("Performance Analysis with Knowledge Transfer")
    plt.legend(title="Models")
    plt.show()
