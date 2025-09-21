import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gym
from abo_3_using_bert import*
from gym import spaces
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For FLOPs calculation
from thop import profile, clever_format
import ptflops

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DQN Network (same architecture as during training)
class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Forward pass"""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

# Function to calculate FLOPs and parameters
def calculate_complexity(model, input_size):
    """Calculate FLOPs, parameters, and MSIP for different input sizes"""
    
    results = []
    
    for size in input_size:
        # Create dummy input
        dummy_input = torch.randn(1, size).to(device)
        
        # Calculate FLOPs and parameters using thop
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # Convert MACs to FLOPs
        
        # Calculate MSIP (Million Stable Isomers Per second)
        # This is a hypothetical metric for our application
        msip = (flops / 1e6) / 1000  # Simplified calculation
        
        results.append({
            'input_size': size,
            'flops': flops,
            'macs': macs,
            'params': params,
            'msip': msip
        })
        
        print(f"Input size: {size}")
        print(f"  FLOPs: {flops/1e6:.2f} MFLOPs")
        print(f"  MACs: {macs/1e6:.2f} MMACs")
        print(f"  Parameters: {params/1e3:.2f} K")
        print(f"  MSIP: {msip:.2f}")
        print("-" * 40)
    
    return results

# Function to plot complexity analysis
def plot_complexity_analysis(results):
    """Plot FLOPs, Parameters, and MSIP vs Input Size"""
    
    # Extract data
    input_sizes = [r['input_size'] for r in results]
    flops = [r['flops']/1e6 for r in results]  # Convert to MFLOPs
    params = [r['params']/1e3 for r in results]  # Convert to K parameters
    msip = [r['msip'] for r in results]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: FLOPs vs Input Size
    ax1.plot(input_sizes, flops, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('FLOPs (Million)')
    ax1.set_title('Computational Complexity (FLOPs)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameters vs Input Size
    ax2.plot(input_sizes, params, 's-', color='#A23B72', linewidth=2, markersize=8)
    ax2.set_xlabel('Input Size')
    ax2.set_ylabel('Parameters (Thousand)')
    ax2.set_title('Model Parameters')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MSIP vs Input Size
    ax3.plot(input_sizes, msip, '^-', color='#F18F01', linewidth=2, markersize=8)
    ax3.set_xlabel('Input Size')
    ax3.set_ylabel('MSIP (Million Stable Isomers/s)')
    ax3.set_title('Performance Metric (MSIP)')
    ax3.grid(True, alpha=0.3)
    
    # Add titles below each subplot
    fig.text(0.16, 0.02, '(a) FLOPs vs Input Size', ha='center', fontsize=14)
    fig.text(0.5, 0.02, '(b) Parameters vs Input Size', ha='center', fontsize=14)
    fig.text(0.84, 0.02, '(c) MSIP vs Input Size', ha='center', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save plots
    plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('complexity_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Function to print detailed table
def print_complexity_table(results):
    """Print detailed complexity analysis table"""
    
    print("\n" + "="*80)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("="*80)
    print(f"{'Input Size':<12} {'FLOPs (M)':<12} {'MACs (M)':<12} {'Params (K)':<12} {'MSIP':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['input_size']:<12} {result['flops']/1e6:<12.2f} {result['macs']/1e6:<12.2f} "
              f"{result['params']/1e3:<12.2f} {result['msip']:<12.2f}")
    
    print("="*80)

def load_and_preprocess_data(file_path):
    """Load and preprocess the perovskite dataset"""
    df = pd.read_csv(file_path)
    numeric_columns = [
        "Radius A [ang]", "Radius B [ang]", "Formation energy [eV/atom]",
        "Stability [eV/atom]", "Band gap [eV]", "Magnetic moment [mu_B]",
        "a", "b", "c", "alpha", "beta", "gamma",
        "Vacancy energy [eV/O atom]", "Volume per atom [A^3/atom]"
    ]
    numerical_cols = [col for col in numeric_columns if col in df.columns]
    categorical_cols = ["A", "B", "Valence A", "Valence B", "Lowest distortion"]
    return df, numerical_cols, categorical_cols

# Main execution
if __name__ == "__main__":
    # Load and preprocess data to get state size
    print("Loading data to determine state size...")
    df, numerical_cols, categorical_cols = load_and_preprocess_data('DFT_ABO₃.csv')
    
    # Create temporary environment to get state size
    stability_predictor = StabilityPredictor()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    env = ABO3Env(train_df, numerical_cols, categorical_cols, stability_predictor)
    
    state_size = env.state_size
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Load pretrained model
    print("Loading pretrained model...")
    model = DQN(state_size, action_size).to(device)
    
    try:
        model.load_state_dict(torch.load('final_model.pth', map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Pretrained model not found. Using randomly initialized model.")
    
    model.eval()
    
    # Define different input sizes to test
    input_sizes = [10, 50, 100, state_size, 200, 300]
    
    # Calculate complexity metrics
    print("\nCalculating computational complexity...")
    results = calculate_complexity(model, input_sizes)
    
    # Print detailed table
    print_complexity_table(results)
    
    # Plot results
    print("\nGenerating complexity analysis plots...")
    plot_complexity_analysis(results)
    
    # Additional analysis for the actual state size
    actual_result = next(r for r in results if r['input_size'] == state_size)
    print(f"\nANALYSIS FOR ACTUAL STATE SIZE ({state_size}):")
    print(f"• FLOPs: {actual_result['flops']/1e6:.2f} Million")
    print(f"• MACs: {actual_result['macs']/1e6:.2f} Million") 
    print(f"• Parameters: {actual_result['params']/1e3:.2f} Thousand")
    print(f"• MSIP: {actual_result['msip']:.2f} Million Stable Isomers/s")
    
    # Estimate inference time (hypothetical)
    # Assuming 100 GFLOPS capability (reasonable for modern CPUs)
    flops_per_second = 100e9  # 100 GFLOPS
    inference_time = (actual_result['flops'] / flops_per_second) * 1000  # in milliseconds
    
    print(f"• Estimated inference time: {inference_time:.4f} ms (at 100 GFLOPS)")
    print(f"• Throughput: {1000/inference_time:.2f} predictions/second")
    
    print("\nComplexity analysis completed!")