import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gym
from gym import spaces
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
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
    # For numerical features, impute with mean
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

# Stability Predictor
class StabilityPredictor:
    """Predicts stability from material features"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=seed)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X, y):
        """Train the stability predictor"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X):
        """Predict stability"""
        if not self.is_trained:
            return 0.5  # Default value

        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return max(0, self.model.predict(X_scaled)[0])  # Ensure non-negative

# Simplified Minstrel client for material suggestions
class MinstrelClient:
    """Client for Minstrel API to get material suggestions"""

    def __init__(self):
        pass

    def get_suggestions(self, current_material, property_of_interest="stability", n_suggestions=5):
        """Get material suggestions from Minstrel (simulated)"""

        suggestions = []
        elements = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Sr', 'Ba',
                   'Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta',
                   'Cr', 'Mo', 'W', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

        for _ in range(n_suggestions):
            suggestion = current_material.copy()

            # Randomly modify some properties
            if random.random() < 0.3:
                suggestion['A'] = random.choice(elements)
                suggestion['Valence A'] = random.randint(1, 3)
                suggestion['Radius A [ang]'] = random.uniform(0.5, 2.0)

            if random.random() < 0.3:
                suggestion['B'] = random.choice(elements)
                suggestion['Valence B'] = random.randint(3, 6)
                suggestion['Radius B [ang]'] = random.uniform(0.3, 1.5)

            if random.random() < 0.2:
                suggestion['Lowest distortion'] = random.choice(['cubic', 'tetragonal', 'orthorhombic', 'rhombohedral'])

            # Add some noise to numerical properties
            for col in ['Formation energy [eV/atom]', 'Band gap [eV]', 'Magnetic moment [mu_B]']:
                if col in suggestion:
                    suggestion[col] += random.uniform(-0.1, 0.1)

            suggestions.append(suggestion)

        return suggestions

# Replay Buffer
class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, buffer_size, batch_size, state_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.state_size = state_size
        self.experience = namedtuple("Experience",
                                    field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        if len(state) != self.state_size:
            state = np.zeros(self.state_size)
        if len(next_state) != self.state_size:
            next_state = np.zeros(self.state_size)

        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        """Sample batch of experiences"""
        if len(self.buffer) < self.batch_size:
            return None

        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

# DQN Network
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

# DQN Agent
class DQNAgent:
    """DQN Agent with Minstrel augmentation"""

    def __init__(self, state_size, action_size, minstrel_client, stability_predictor):
        self.state_size = state_size
        self.action_size = action_size
        self.minstrel_client = minstrel_client
        self.stability_predictor = stability_predictor

        # Q-Network and target network
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.0005)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size=50000, batch_size=128, state_size=state_size)

        # Training parameters
        self.t_step = 0
        self.update_every = 4
        self.gamma = 0.99
        self.tau = 1e-3

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory and learn"""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            experiences = self.memory.sample()
            if experiences is not None:
                loss = self.learn(experiences)
                return loss
        return None

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy"""
        if len(state) != self.state_size:
            state = np.zeros(self.state_size)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        return loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def augment_with_minstrel(self, state, n_suggestions=2):
        """Augment replay buffer with Minstrel suggestions"""
        material_representation = {
            'A': 'Element', 'B': 'Element', 'Lowest distortion': 'cubic'
        }

        # Add numerical features
        for i, col in enumerate(['Valence A', 'Valence B', 'Radius A [ang]', 'Radius B [ang]',
                               'Formation energy [eV/atom]', 'Band gap [eV]', 'Magnetic moment [mu_B]']):
            if i < len(state):
                material_representation[col] = state[i]

        suggestions = self.minstrel_client.get_suggestions(material_representation, n_suggestions=n_suggestions)

        for suggestion in suggestions:
            suggested_state = np.array([
                suggestion['Valence A'], suggestion['Valence B'],
                suggestion['Radius A [ang]'], suggestion['Radius B [ang]'],
                suggestion['Formation energy [eV/atom]'], suggestion['Band gap [eV]'],
                suggestion['Magnetic moment [mu_B]']
            ])

            # Pad to match state size
            if len(suggested_state) < self.state_size:
                suggested_state = np.pad(suggested_state, (0, self.state_size - len(suggested_state)))

            # Predict stability and calculate reward
            estimated_stability = self.stability_predictor.predict(suggested_state)
            estimated_reward = self._calculate_reward(estimated_stability)

            random_action = random.randint(0, self.action_size - 1)
            self.memory.add(state, random_action, estimated_reward, suggested_state, False)

    def _calculate_reward(self, stability):
        """Calculate reward with proper scaling and bonuses"""
        # Base reward (higher for lower stability values)
        reward = 10.0 / (stability + 0.1)  # Scale to make rewards more meaningful

        # Bonus for very stable compounds
        if stability < 0.1:
            reward += 20.0
        elif stability < 0.5:
            reward += 10.0

        return reward

# Gym Environment
class ABO3Env(gym.Env):
    """Custom Gym environment for ABO3 perovskite design"""

    def __init__(self, df, numerical_cols, categorical_cols, stability_predictor, target_col="Stability [eV/atom]"):
        super(ABO3Env, self).__init__()

        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.stability_predictor = stability_predictor

        self._preprocess_data()

        # Action space: mutate different features
        self.action_space = spaces.Discrete(10)

        # Observation space
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.state_size,), dtype=np.float32
        )

        self.reset()

    def _preprocess_data(self):
        """Preprocess the data for RL environment"""
        self.scaler = StandardScaler()
        numerical_data = self.scaler.fit_transform(self.df[self.numerical_cols])

        categorical_data = pd.get_dummies(self.df[self.categorical_cols])

        self.features = np.hstack([numerical_data, categorical_data.values])
        self.feature_columns = list(self.numerical_cols) + list(categorical_data.columns)
        self.state_size = self.features.shape[1]

        self.targets = self.df[self.target_col].values
        self.original_indices = self.df.index.values

    def reset(self):
        """Reset to a random initial state"""
        self.current_idx = random.randint(0, len(self.features) - 1)
        self.current_state = self.features[self.current_idx].copy()
        self.current_stability = self.targets[self.current_idx]
        self.steps = 0

        return self.current_state.copy()

    def step(self, action):
        """Take an action"""
        self.steps += 1
        new_state = self.current_state.copy()

        # Define mutation parameters
        mutation_params = [
            (0, 0.3), (1, 0.3), (2, 0.1), (3, 0.1),
            (4, 0.2), (5, 0.4), (6, 0.2), (7, 0.1), (8, 0.1)
        ]

        if action < 9:  # Feature mutation
            feature_idx, mutation_range = mutation_params[action]
            if feature_idx < len(new_state):
                new_state[feature_idx] += random.uniform(-mutation_range, mutation_range)
        else:  # Reset to known compound
            new_idx = random.randint(0, len(self.features) - 1)
            new_state = self.features[new_idx].copy()

        # Clip values
        new_state = np.clip(new_state, -3.0, 3.0)

        # Predict stability
        estimated_stability = self.stability_predictor.predict(new_state)

        # Calculate reward with proper scaling
        reward = self._calculate_reward(estimated_stability)

        # Update state
        self.current_state = new_state
        self.current_stability = estimated_stability

        # Episode termination
        done = self.steps >= 30 or estimated_stability < 0.05 or random.random() < 0.05

        return new_state, reward, done, {"stability": estimated_stability}

    def _calculate_reward(self, stability):
        """Calculate properly scaled reward"""
        # Positive rewards for good stability
        reward = 15.0 / (stability + 0.01)  # Strong positive scaling

        # Additional bonuses
        if stability < 0.05:
            reward += 50.0  # Big bonus for excellent stability
        elif stability < 0.1:
            reward += 25.0
        elif stability < 0.2:
            reward += 10.0

        return reward

    def render(self, mode='human'):
        print(f"Stability: {self.current_stability:.4f} eV/atom, Reward: {self._calculate_reward(self.current_stability):.2f}")

# Training function with tracking for research plots
def train_dqn(env, agent, n_episodes=2000, max_t=30, eps_start=1.0, eps_end=0.01, eps_decay=0.998):
    """Deep Q-Learning training with tracking for research plots"""

    scores = []
    scores_window = deque(maxlen=100)
    losses = []
    eps = eps_start
    
    # Track best stability per episode and episode-wise stability trends
    best_stability_per_episode = []
    episode_stability_trends = []
    all_rewards = []  # Track all rewards for distribution analysis
    all_states = []   # Track all states for dimensionality reduction

    progress_bar = tqdm(range(1, n_episodes + 1), desc="Training")

    for i_episode in progress_bar:
        state = env.reset()
        score = 0
        episode_losses = []
        
        # Track stability trend within this episode
        episode_stabilities = [env.current_stability]
        best_episode_stability = env.current_stability
        episode_rewards = []
        episode_states = [state.copy()]

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)

            if loss is not None:
                episode_losses.append(loss)

            if random.random() < 0.1:
                agent.augment_with_minstrel(state, n_suggestions=1)

            state = next_state
            score += reward
            episode_rewards.append(reward)
            episode_states.append(state.copy())
            
            # Track stability
            current_stability = info['stability']
            episode_stabilities.append(current_stability)
            best_episode_stability = min(best_episode_stability, current_stability)

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        best_stability_per_episode.append(best_episode_stability)
        episode_stability_trends.append(episode_stabilities)
        all_rewards.extend(episode_rewards)
        all_states.extend(episode_states)

        if episode_losses:
            avg_loss = np.mean(episode_losses)
            losses.append(avg_loss)

        eps = max(eps_end, eps_decay * eps)

        progress_bar.set_postfix({
            'AvgScore': f'{np.mean(scores_window):.2f}',
            'Eps': f'{eps:.3f}',
            'BestStability': f'{min(best_stability_per_episode):.3f}'
        })

        if i_episode % 500 == 0:
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_episode_{i_episode}.pth')

    return scores, losses, best_stability_per_episode, episode_stability_trends, all_rewards, all_states

# Evaluation function
def evaluate_agent(env, agent, n_episodes=20):
    """Evaluate the trained agent"""
    scores = []
    stabilities_found = []
    evaluation_rewards = []
    evaluation_states = []

    for i_episode in range(n_episodes):
        state = env.reset()
        score = 0
        episode_stabilities = []
        episode_rewards = []
        episode_states = [state.copy()]

        for t in range(50):  # Longer evaluation episodes
            action = agent.act(state, eps=0.0)
            next_state, reward, done, info = env.step(action)

            state = next_state
            score += reward
            episode_stabilities.append(info['stability'])
            episode_rewards.append(reward)
            episode_states.append(state.copy())

            if done:
                break

        scores.append(score)
        stabilities_found.extend(episode_stabilities)
        evaluation_rewards.extend(episode_rewards)
        evaluation_states.extend(episode_states)

    return scores, stabilities_found, evaluation_rewards, evaluation_states

# Function to create dimensionality reduction plots
def create_dimensionality_reduction_plots(states, rewards, method='tsne'):
    """Create t-SNE or PCA plots of state space"""

    # Convert to numpy arrays
    states_array = np.array(states)
    rewards_array = np.array(rewards)

    # Ensure both arrays have the same length
    min_len = min(len(states_array), len(rewards_array))
    states_array = states_array[:min_len]
    rewards_array = rewards_array[:min_len]

    # Sample a subset of points if too many
    n_samples = min(5000, min_len)
    if min_len > n_samples:
        indices = np.random.choice(min_len, n_samples, replace=False)
        states_sample = states_array[indices]
        rewards_sample = rewards_array[indices]
    else:
        states_sample = states_array
        rewards_sample = rewards_array
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
        reduced_states = reducer.fit_transform(states_sample)
        method_name = 't-SNE'
    else:  # PCA
        reducer = PCA(n_components=2, random_state=seed)
        reduced_states = reducer.fit_transform(states_sample)
        method_name = 'PCA'
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_states[:, 0], reduced_states[:, 1], 
                         c=rewards_sample, cmap='viridis', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Reward')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.title(f'{method_name} Projection of State Space Colored by Reward')
    
    # Save plot
    plt.savefig(f'{method_name.lower()}_projection.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{method_name.lower()}_projection.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{method_name.lower()}_projection.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    return reduced_states

# Function to create reward distribution plots
def create_reward_distribution_plots(train_rewards, eval_rewards):
    """Create plots showing reward distribution"""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Histogram of rewards
    ax1.hist(train_rewards, bins=50, alpha=0.7, color='#2E86AB', label='Training')
    ax1.hist(eval_rewards, bins=50, alpha=0.7, color='#A23B72', label='Evaluation')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    # Plot 2: Cumulative distribution of rewards
    sorted_train = np.sort(train_rewards)
    sorted_eval = np.sort(eval_rewards)
    
    cdf_train = np.arange(1, len(sorted_train) + 1) / len(sorted_train)
    cdf_eval = np.arange(1, len(sorted_eval) + 1) / len(sorted_eval)
    
    ax2.plot(sorted_train, cdf_train, color='#2E86AB', linewidth=2, label='Training')
    ax2.plot(sorted_eval, cdf_eval, color='#A23B72', linewidth=2, label='Evaluation')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF of Reward Values')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    # Add titles below each subplot
    fig.text(0.25, 0.02, '(a) Reward Distribution', ha='center', fontsize=14)
    fig.text(0.75, 0.02, '(b) CDF of Reward Values', ha='center', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('reward_distribution.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig('reward_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print reward statistics
    print("\n=== REWARD STATISTICS ===")
    print(f"Training rewards - Mean: {np.mean(train_rewards):.2f}, Max: {np.max(train_rewards):.2f}, Min: {np.min(train_rewards):.2f}")
    print(f"Evaluation rewards - Mean: {np.mean(eval_rewards):.2f}, Max: {np.max(eval_rewards):.2f}, Min: {np.min(eval_rewards):.2f}")

# Function to create research paper plots with titles below
def create_research_plots(best_stability_per_episode, episode_stability_trends, test_stabilities, test_env):
    """Create publication-quality plots for research paper with titles below"""
    
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
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Best Stability per Episode
    ax1.plot(best_stability_per_episode, color='#2E86AB', linewidth=1.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Best Stability [eV/atom]')
    ax1.set_yscale('log')
    
    # Add a smoothed trend line
    window_size = 50
    if len(best_stability_per_episode) > window_size:
        smoothed = np.convolve(best_stability_per_episode, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(best_stability_per_episode)), smoothed, 
                color='#A23B72', linewidth=2, label=f'{window_size}-episode moving average')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    
    # Plot 2: Cumulative Distribution Function (CDF) of Stability
    # Sort the stabilities for CDF
    sorted_test_stabilities = np.sort(test_env.targets)
    sorted_agent_stabilities = np.sort(test_stabilities)
    
    # Calculate CDF values
    cdf_test = np.arange(1, len(sorted_test_stabilities) + 1) / len(sorted_test_stabilities)
    cdf_agent = np.arange(1, len(sorted_agent_stabilities) + 1) / len(sorted_agent_stabilities)
    
    # Plot CDFs
    ax2.plot(sorted_test_stabilities, cdf_test, color='#4B3F72', linewidth=2, label='Test Set')
    ax2.plot(sorted_agent_stabilities, cdf_agent, color='#FFC857', linewidth=2, label='Agent Discovered')
    
    ax2.set_xlabel('Stability [eV/atom]')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    # Plot 3: Episode-wise Stability Trend (sample a few episodes)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Stability [eV/atom]')
    ax3.set_yscale('log')
    
    # Sample a few episodes to show trends
    sample_episodes = [0, len(episode_stability_trends)//4, len(episode_stability_trends)//2, 
                      len(episode_stability_trends)*3//4, len(episode_stability_trends)-1]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4B3F72']
    
    for i, ep_idx in enumerate(sample_episodes):
        if ep_idx < len(episode_stability_trends):
            stability_trend = episode_stability_trends[ep_idx]
            ax3.plot(range(len(stability_trend)), stability_trend, 
                    color=colors[i], linewidth=1.5, alpha=0.8, label=f'Episode {ep_idx+1}')
    
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    # Add titles below each subplot
    fig.text(0.16, 0.02, '(a) Best Stability per Episode', ha='center', fontsize=14)
    fig.text(0.5, 0.02, '(b) CDF of Stability Values', ha='center', fontsize=14)
    fig.text(0.84, 0.02, '(c) Stability Trend Within Episodes', ha='center', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for titles at bottom
    plt.savefig('research_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig('research_plots.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig('research_plots.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics for the paper
    print("\n=== RESEARCH PAPER STATISTICS ===")
    print(f"Best stability discovered: {np.min(test_stabilities):.4f} eV/atom")
    print(f"Average stability discovered: {np.mean(test_stabilities):.4f} eV/atom")
    print(f"Median stability discovered: {np.median(test_stabilities):.4f} eV/atom")
    print(f"Improvement over test set best: {np.min(test_env.targets) - np.min(test_stabilities):.4f} eV/atom")
    
    # Calculate percentage of compounds with stability < threshold
    threshold = 0.1
    agent_below_threshold = np.sum(np.array(test_stabilities) < threshold) / len(test_stabilities) * 100
    test_below_threshold = np.sum(test_env.targets < threshold) / len(test_env.targets) * 100
    
    print(f"Compounds with stability < {threshold} eV/atom:")
    print(f"  - Agent discovered: {agent_below_threshold:.1f}%")
    print(f"  - Test set: {test_below_threshold:.1f}%")

# Main execution
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df, numerical_cols, categorical_cols = load_and_preprocess_data('DFT_ABO₃.csv')

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    # Train stability predictor
    print("Training stability predictor...")
    stability_predictor = StabilityPredictor()

    # Create temp env for training data
    temp_env = ABO3Env(train_df, numerical_cols, categorical_cols, StabilityPredictor())
    X_train = temp_env.features
    y_train = temp_env.targets
    stability_predictor.train(X_train, y_train)

    # Create environments
    train_env = ABO3Env(train_df, numerical_cols, categorical_cols, stability_predictor)
    test_env = ABO3Env(test_df, numerical_cols, categorical_cols, stability_predictor)

    print(f"State size: {train_env.state_size}, Action size: {train_env.action_space.n}")

    # Initialize components
    minstrel_client = MinstrelClient()
    agent = DQNAgent(train_env.state_size, train_env.action_space.n, minstrel_client, stability_predictor)

    # Train with tracking for research plots
    print("Training DQN agent with research tracking...")
    scores, losses, best_stability_per_episode, episode_stability_trends, train_rewards, train_states = train_dqn(train_env, agent, n_episodes=2000)

    # Evaluate
    print("Evaluating agent...")
    test_scores, test_stabilities, eval_rewards, eval_states = evaluate_agent(test_env, agent)

    # Create research paper plots
    print("Creating research paper plots...")
    create_research_plots(best_stability_per_episode, episode_stability_trends, test_stabilities, test_env)
    
    # Create dimensionality reduction plots
    print("Creating dimensionality reduction plots...")
    # Combine training and evaluation states for a comprehensive view
    all_states = train_states + eval_states
    all_rewards = train_rewards + eval_rewards
    
    # Create t-SNE projection
    create_dimensionality_reduction_plots(all_states, all_rewards, method='tsne')
    
    # Create PCA projection
    create_dimensionality_reduction_plots(all_states, all_rewards, method='pca')
    
    # Create reward distribution plots
    print("Creating reward distribution plots...")
    create_reward_distribution_plots(train_rewards, eval_rewards)

    # Print results
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(test_scores):.2f}")
    print(f"Best Stability Found: {np.min(test_stabilities):.4f} eV/atom")
    print(f"Average Stability Found: {np.mean(test_stabilities):.4f} eV/atom")
    print(f"Worst Stability Found: {np.max(test_stabilities):.4f} eV/atom")

    # Compare with dataset
    print(f"\nDataset Comparison:")
    print(f"Best stability in test set: {np.min(test_env.targets):.4f} eV/atom")
    print(f"Average stability in test set: {np.mean(test_env.targets):.4f} eV/atom")

    torch.save(agent.qnetwork_local.state_dict(), 'final_model.pth')
    print("Training completed and all plots saved!")