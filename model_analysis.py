import torch
import torch.nn as nn
import os
import pandas as pd
from torchviz import make_dot
import matplotlib.pyplot as plt

# Define the models to analyze
model_files = ["ppo_model.pth", "dqn_model.pth"]  # Add more model files as needed

# Define a dictionary to store model performance metrics
model_performance = {
    "Model": [],
    "Number of Parameters": [],
    "Layers": [],
    "File Size (MB)": [],
}

# Function to visualize the model hierarchy using torchviz
def visualize_model(model, model_name):
    # Create a random input tensor for visualization purposes (adjust shape as needed)
    dummy_input = torch.randn(4, 3)  # Assuming input dimension is 4 for this example
    output = model(dummy_input)
    
    # Create a visualization of the model hierarchy
    make_dot(output.mean(), params=dict(model.named_parameters())).render(f"{model_name}_model_hierarchy", format="png")
    print(f"Model hierarchy for {model_name} saved as {model_name}_model_hierarchy.png")

# Function to load a PyTorch model from a file and return it
def load_model(file_path):
    # Load the state_dict from the file
    state_dict = torch.load(file_path)
    
    # Create a dummy model based on the file name (you may need to adjust this based on your actual models)
    if "ppo" in file_path:
        # Assuming PPO uses an ActorCritic model with input size 4 and output size 3
        model = ActorCritic(4, 3)
    elif "dqn" in file_path:
        # Assuming DQN uses a simple feed-forward network with input size 4 and output size 3
        model = DQN(4, 3)
    
    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    
    return model

# Function to get the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to analyze each model and add its metrics to the dictionary
def analyze_models():
    for file in model_files:
        if os.path.exists(file):
            print(f"Analyzing {file}...")
            
            # Load the model from file
            model = load_model(file)
            
            # Get the number of parameters in the model
            num_params = count_parameters(model)
            
            # Get the number of layers (assuming each layer has named parameters)
            num_layers = len(list(model.named_parameters()))
            
            # Get the file size in MB
            file_size = os.path.getsize(file) / (1024 * 1024)  # Convert bytes to MB
            
            # Visualize the model hierarchy and save it as an image
            visualize_model(model, os.path.splitext(file)[0])
            
            # Append data to the performance dictionary
            model_performance["Model"].append(os.path.splitext(file)[0])
            model_performance["Number of Parameters"].append(num_params)
            model_performance["Layers"].append(num_layers)
            model_performance["File Size (MB)"].append(round(file_size, 2))
        else:
            print(f"File {file} not found.")

# Function to display performance comparison table using pandas DataFrame
def display_comparison_table():
    df = pd.DataFrame(model_performance)
    print("\nModel Performance Comparison:")
    print(df)

# Define your PPO and DQN models here (simplified versions for this example)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Main function to run the analysis and display results
if __name__ == "__main__":
    analyze_models()
    display_comparison_table()