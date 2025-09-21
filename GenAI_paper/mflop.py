import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DQN Network 
class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(DQN, self).__init__()
        self.state_size = state_size
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

# Manual FLOPs calculation function
def calculate_flops_manual(model, input_size):
    """Manually calculate FLOPs for DQN network"""
    total_macs = 0
    
    # Input layer: state_size -> 128
    total_macs += input_size * 128  # MACs for linear layer (matrix multiplication)
    
    # Hidden layer: 128 -> 64  
    total_macs += 128 * 64  # MACs for linear layer
    
    # Output layer: 64 -> 10
    total_macs += 64 * 10  # MACs for linear layer
    
    # Add bias operations (each bias add is 1 FLOP per output element)
    total_macs += 128 + 64 + 10  # Bias additions
    
    # ReLU activations (1 FLOP per element)
    total_macs += 128 + 64  # ReLU operations
    
    total_flops = total_macs * 2  # Convert MACs to FLOPs (1 MAC = 2 FLOPs)
    params = sum(p.numel() for p in model.parameters())
    
    return total_flops, total_macs, params

# Function to calculate complexity metrics
def calculate_complexity_corrected(action_size=10, hidden_layers=[128, 64]):
    """Corrected FLOPs calculation for different state sizes"""
    
    results = []
    input_sizes = [10, 50, 100, 173, 200, 300]
    
    for state_size in input_sizes:
        # Create model
        model = DQN(state_size, action_size, hidden_layers).to(device)
        model.eval()
        
        # Calculate using manual method
        flops, macs, params = calculate_flops_manual(model, state_size)
        
        msip = 1000 / (flops / 1e9)  # predictions per second at 1 GFLOPS, scaled to millions
        
        results.append({
            'state_size': state_size,
            'flops': flops,
            'macs': macs,
            'params': params,
            'msip': msip
        })
        
        print(f"State size: {state_size}")
        print(f"  FLOPs: {flops/1e6:.2f} MFLOPs")
        print(f"  MACs: {macs/1e6:.2f} MMACs")
        print(f"  Parameters: {params/1e3:.2f} K")
        print(f"  MSIP: {msip:.2f}")
        print("-" * 40)
    
    return results

# Function to plot complexity analysis
def plot_complexity_analysis(results):
    """Plot FLOPs, Parameters, and MSIP vs State Size"""
    
    # Set style for research paper
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (15, 5),
        'figure.dpi': 300,
    })
    
    # Extract data
    state_sizes = [r['state_size'] for r in results]
    flops = [r['flops']/1e6 for r in results]  # Convert to MFLOPs
    params = [r['params']/1e3 for r in results]  # Convert to K parameters
    msip = [r['msip'] for r in results]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: FLOPs vs State Size
    ax1.plot(state_sizes, flops, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax1.set_xlabel('State Size')
    ax1.set_ylabel('FLOPs (Million)')
    ax1.set_title('Computational Complexity (FLOPs)')
    # ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameters vs State Size
    ax2.plot(state_sizes, params, 's-', color='#A23B72', linewidth=2, markersize=8)
    ax2.set_xlabel('State Size')
    ax2.set_ylabel('Parameters (Thousand)')
    ax2.set_title('Model Parameters')
    # ax2.grid(True, alpha=0.3)
    
    # Plot 3: MSIP vs State Size
    ax3.plot(state_sizes, msip, '^-', color='#F18F01', linewidth=2, markersize=8)
    ax3.set_xlabel('State Size')
    ax3.set_ylabel('MSIP (Million Stable Isomers/s)')
    ax3.set_title('Performance Metric (MSIP)')
    # ax3.grid(True, alpha=0.3)
    
    # Add titles below each subplot
    fig.text(0.16, 0.02, '(a) FLOPs vs State Size', ha='center', fontsize=14)
    fig.text(0.5, 0.02, '(b) Parameters vs State Size', ha='center', fontsize=14)
    fig.text(0.84, 0.02, '(c) MSIP vs State Size', ha='center', fontsize=14)
    
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
    print(f"{'State Size':<12} {'FLOPs (M)':<12} {'MACs (M)':<12} {'Params (K)':<12} {'MSIP':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['state_size']:<12} {result['flops']/1e6:<12.2f} {result['macs']/1e6:<12.2f} "
              f"{result['params']/1e3:<12.2f} {result['msip']:<12.2f}")
    
    print("="*80)

# Function to analyze performance for actual model
def analyze_performance(results, target_size=173, flops_capability=100e9):
    """Analyze performance metrics for the target state size"""
    
    target_result = next(r for r in results if r['state_size'] == target_size)
    
    print(f"\nPERFORMANCE ANALYSIS FOR STATE SIZE {target_size}:")
    print("="*50)
    
    flops = target_result['flops']
    inference_time = (flops / flops_capability) * 1000  # in milliseconds
    throughput = 1000 / inference_time  # predictions per second
    
    print(f"• FLOPs per inference: {flops/1e6:.2f} MFLOPs")
    print(f"• Parameters: {target_result['params']/1e3:.2f} K")
    print(f"• Inference time: {inference_time:.4f} ms (at {flops_capability/1e9:.0f} GFLOPS)")
    print(f"• Throughput: {throughput:,.0f} predictions/second")
    print(f"• MSIP: {target_result['msip']:.2f} Million Stable Isomers/s")
    
    # Calculate for different hardware capabilities
    print(f"\nSCALING ACROSS HARDWARE PLATFORMS:")
    print("-" * 40)
    capabilities = [10e9, 50e9, 100e9, 200e9]  # 10, 50, 100, 200 GFLOPS
    
    for cap in capabilities:
        itime = (flops / cap) * 1000
        tput = 1000 / itime
        print(f"• At {cap/1e9:.0f} GFLOPS: {tput:,.0f} predictions/s ({itime:.4f} ms/inf)")

# Main execution
if __name__ == "__main__":
    print("CORRECTED COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("="*60)
    
    # Calculate complexity with corrected method
    results = calculate_complexity_corrected(action_size=10, hidden_layers=[128, 64])
    
    # Print detailed table
    print_complexity_table(results)
    
    # Plot results
    print("\nGenerating complexity analysis plots...")
    plot_complexity_analysis(results)
    
    # Analyze performance
    analyze_performance(results, target_size=173, flops_capability=100e9)
    
    print("\nAnalysis completed successfully!")