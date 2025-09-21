import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set style for publication-quality figure
plt.style.use('default')
sns.set_palette("deep")

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset (same as training)
def load_and_preprocess_data(file_path):
    """Load and preprocess the perovskite dataset"""
    df = pd.read_csv(file_path)

    # Convert numeric-like columns from object to float
    numeric_columns = [
        "Radius A [ang]", "Radius B [ang]", "Formation energy [eV/atom]",
        "Stability [eV/atom]", "Band gap [eV]", "Magnetic moment [mu_B]",
        "a", "b", "c", "alpha", "beta", "gamma",
        "Vacancy energy [eV/O atom]", "Volume per atom [A^3/atom]"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_cols = [col for col in numeric_columns if col in df.columns]
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

    # For categorical features, impute with most frequent
    categorical_cols = ["A", "B", "Valence A", "Valence B", "Lowest distortion"]
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    for col in categorical_cols:
        if col in df.columns:
            df[col] = categorical_imputer.fit_transform(df[[col]]).ravel()

    return df, numerical_cols, categorical_cols

# DQN Network (same as training)
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
            x = torch.relu(layer(x))
        return self.output_layer(x)

# Load the data
print("Loading and preprocessing data...")
df, numerical_cols, categorical_cols = load_and_preprocess_data('DFT_ABO₃.csv')

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

try:
    # Load the state dict to check the original shape
    state_dict = torch.load('final_model.pth', map_location=device)
    
    # The first layer weight shape will be [128, original_state_size]
    original_state_size = state_dict['layers.0.weight'].shape[1]
    print(f"Original state size from saved model: {original_state_size}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using current data preprocessing...")
    original_state_size = None

# Prepare data with the same preprocessing
def prepare_data(df, numerical_cols, categorical_cols, original_state_size=None, is_training=False):
    """Prepare data for evaluation"""
    # Scale numerical features
    if is_training or not os.path.exists('scaler.joblib'):
        scaler = StandardScaler()
        numerical_data = scaler.fit_transform(df[numerical_cols])
        if is_training:
            joblib.dump(scaler, 'scaler.joblib')
    else:
        scaler = joblib.load('scaler.joblib')
        numerical_data = scaler.transform(df[numerical_cols])
    
    # One-hot encode categorical features
    categorical_data = pd.get_dummies(df[categorical_cols])
    
    if is_training or not os.path.exists('categorical_columns.joblib'):
        training_categorical_columns = categorical_data.columns
        if is_training:
            joblib.dump(training_categorical_columns, 'categorical_columns.joblib')
    else:
        training_categorical_columns = joblib.load('categorical_columns.joblib')
        
        # Ensure we have the same columns as training
        for col in training_categorical_columns:
            if col not in categorical_data.columns:
                categorical_data[col] = 0
                
        # Reorder columns to match training
        categorical_data = categorical_data[training_categorical_columns]
    
    # Combine features
    features = np.hstack([numerical_data, categorical_data.values])
    targets = df["Stability [eV/atom]"].values
    
    # If we know the original state size, pad or truncate features
    if original_state_size is not None:
        current_size = features.shape[1]
        if current_size < original_state_size:
            # Pad with zeros
            padding = np.zeros((features.shape[0], original_state_size - current_size))
            features = np.hstack([features, padding])
            print(f"Padded features from {current_size} to {original_state_size} dimensions")
        elif current_size > original_state_size:
            # Truncate
            features = features[:, :original_state_size]
            print(f"Truncated features from {current_size} to {original_state_size} dimensions")
    
    return features, targets

# Prepare test data
X_test, y_test = prepare_data(test_df, numerical_cols, categorical_cols, original_state_size)
state_size = X_test.shape[1]
print(f"Final state size: {state_size}")

# Load the trained model
print("Loading trained model...")
model = DQN(state_size, 10)  # 10 is the action size from your training code
model.load_state_dict(torch.load('final_model.pth', map_location=device))
model.to(device)
model.eval()

# Make predictions
print("Making predictions...")
predictions = []
with torch.no_grad():
    for i in range(len(X_test)):
        state = torch.from_numpy(X_test[i]).float().to(device)
        action_values = model(state)
    
        max_q_value = torch.max(action_values).item()
        predictions.append(max_q_value)

# Convert predictions to numpy array
predictions = np.array(predictions)

# For a better comparison, let's also create a simple baseline model
print("Training baseline Random Forest model for comparison...")
X_train, y_train = prepare_data(train_df, numerical_cols, categorical_cols, original_state_size, is_training=True)
rf_model = RandomForestRegressor(n_estimators=100, random_state=seed)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

min_pred, max_pred = np.min(predictions), np.max(predictions)
min_actual, max_actual = np.min(y_test), np.max(y_test)
scaled_predictions = min_actual + (predictions - min_pred) * (max_actual - min_actual) / (max_pred - min_pred)

# Create the plot
print("Creating plot...")
plt.figure(figsize=(10, 8))

# Create scatter plot for DQN predictions
plt.scatter(y_test, scaled_predictions, alpha=0.6, s=50, color='steelblue', 
            edgecolor='white', linewidth=0.5, label='DQN Predictions')

# Create scatter plot for RF predictions
plt.scatter(y_test, rf_predictions, alpha=0.6, s=50, color='salmon', 
            edgecolor='white', linewidth=0.5, label='RF Predictions')

# Add perfect prediction line
max_val = max(np.max(y_test), np.max(scaled_predictions), np.max(rf_predictions))
min_val = min(np.min(y_test), np.min(scaled_predictions), np.min(rf_predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

# Calculate R-squared for both models
dqn_r2 = r2_score(y_test, scaled_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Add R-squared text
plt.text(0.05, 0.95, f'DQN R² = {dqn_r2:.3f}\nRF R² = {rf_r2:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Customize the plot
plt.xlabel('Actual Stability [eV/atom]', fontsize=14)
plt.ylabel('Predicted Stability [eV/atom]', fontsize=14)
# plt.title('Predicted vs Actual Stability Values', fontsize=16, fontweight='bold')

# Move legend to bottom
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=12)

# Remove grid
plt.grid(False)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

# Save the figure with high DPI for publication
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.savefig('updated_predicted_vs_actual.pdf', bbox_inches='tight')  # For publication

plt.show()

# Print some statistics
print(f"\nModel Performance Statistics:")
print(f"DQN R-squared: {dqn_r2:.4f}")
print(f"DQN Mean Absolute Error: {mean_absolute_error(y_test, scaled_predictions):.4f}")
print(f"DQN Root Mean Square Error: {np.sqrt(mean_squared_error(y_test, scaled_predictions)):.4f}")
print(f"\nRF R-squared: {rf_r2:.4f}")
print(f"RF Mean Absolute Error: {mean_absolute_error(y_test, rf_predictions):.4f}")
print(f"RF Root Mean Square Error: {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")

print("Plot saved as 'predicted_vs_actual.png' and 'predicted_vs_actual.pdf'")