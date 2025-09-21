import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from stqdm import stqdm
import time

# Set page configuration
st.set_page_config(
    page_title="Generative AI--Augmented Reinforcement Learning for Stability Optimization",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .upload-box {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin-bottom: 2rem;
    }
    .stability-good {
        color: #28a745;
        font-weight: bold;
    }
    .stability-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .stability-poor {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# DQN Network Class (must match training architecture)
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
    
    def forward(self, state):
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class StabilityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_mappings = {}
    
    def load_model(self, model_path, feature_columns):
        self.feature_columns = feature_columns
        self.model = DQN(len(feature_columns), 10)  # action_size=10 as in training
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
    
    def predict_stability(self, features):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to tensor and predict
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(features_tensor)
        
        # Convert Q-values to stability (simplified approach)
        # In a real scenario, you might have a separate regression head
        stability = float(torch.sigmoid(q_values.mean()) * 2.0)  # Scale to 0-2 eV range
        return stability

def create_sample_data():
    """Create sample data for demonstration"""
    data = {
        'A': ['Sr', 'Ba', 'Ca', 'La', 'Y'],
        'B': ['Ti', 'Zr', 'Nb', 'Fe', 'Mn'],
        'Valence A': [2, 2, 2, 3, 3],
        'Valence B': [4, 4, 5, 3, 4],
        'Radius A [ang]': [1.44, 1.61, 1.34, 1.36, 1.04],
        'Radius B [ang]': [0.605, 0.72, 0.69, 0.645, 0.53],
        'Formation energy [eV/atom]': [-0.12, -0.08, 0.15, -0.05, 0.25],
        'Band gap [eV]': [3.2, 3.8, 2.1, 2.8, 1.5],
        'Magnetic moment [mu_B]': [0.0, 0.0, 0.5, 3.5, 3.0],
        'a': [3.905, 4.192, 3.821, 3.926, 3.675],
        'b': [3.905, 4.192, 3.821, 3.926, 3.675],
        'c': [3.905, 4.192, 3.821, 3.926, 3.675],
        'alpha': [90.0, 90.0, 90.0, 90.0, 90.0],
        'beta': [90.0, 90.0, 90.0, 90.0, 90.0],
        'gamma': [90.0, 90.0, 90.0, 90.0, 90.0],
        'Lowest distortion': ['cubic', 'cubic', 'tetragonal', 'orthorhombic', 'rhombohedral']
    }
    return pd.DataFrame(data)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚗️ ABO₃ Perovskite Stability Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict stability of ABO₃ compounds using our Minstrel-augmented DQN model")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StabilityPredictor()
        # Load model (in real app, this would load from actual trained weights)
        try:
            feature_columns = [
                'Valence A', 'Valence B', 'Radius A [ang]', 'Radius B [ang]',
                'Formation energy [eV/atom]', 'Band gap [eV]', 'Magnetic moment [mu_B]',
                'a', 'b', 'c', 'alpha', 'beta', 'gamma'
            ]
            st.session_state.predictor.load_model('final_model.pth', feature_columns)
        except:
            st.warning("Model file not found. Using demonstration mode.")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/test-tube.png", width=100)
        st.title("Navigation")
        app_mode = st.radio(
            "Choose Input Method:",
            ["Manual Input", "CSV Upload", "Sample Data"],
            index=0
        )
        
        st.markdown("---")
        st.info("""
        **About this app:**
        - Predicts stability of ABO₃ perovskites
        - Uses Minstrel-augmented DQN model
        - Accepts manual input or CSV files
        - Provides detailed visualizations
        """)
    
    # Main content based on selection
    if app_mode == "Manual Input":
        manual_input_section()
    elif app_mode == "CSV Upload":
        csv_upload_section()
    else:
        sample_data_section()

def manual_input_section():
    st.markdown('<h2 class="sub-header">🔧 Manual Compound Input</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Element Properties")
        a_element = st.selectbox("Element A", ["Sr", "Ba", "Ca", "La", "Y", "K", "Na", "Li"])
        b_element = st.selectbox("Element B", ["Ti", "Zr", "Nb", "Fe", "Mn", "V", "Cr", "Co"])
        valence_a = st.slider("Valence A", 1, 4, 2)
        valence_b = st.slider("Valence B", 3, 6, 4)
        radius_a = st.slider("Radius A [Å]", 0.5, 2.0, 1.44, 0.01)
        radius_b = st.slider("Radius B [Å]", 0.3, 1.5, 0.605, 0.01)
    
    with col2:
        st.markdown("### Physical Properties")
        formation_energy = st.slider("Formation Energy [eV/atom]", -1.0, 1.0, -0.12, 0.01)
        band_gap = st.slider("Band Gap [eV]", 0.0, 6.0, 3.2, 0.1)
        magnetic_moment = st.slider("Magnetic Moment [μB]", 0.0, 5.0, 0.0, 0.1)
        
        st.markdown("### Lattice Parameters")
        col21, col22, col23 = st.columns(3)
        with col21:
            a = st.number_input("a [Å]", 3.0, 5.0, 3.905, 0.001)
        with col22:
            b = st.number_input("b [Å]", 3.0, 5.0, 3.905, 0.001)
        with col23:
            c = st.number_input("c [Å]", 3.0, 5.0, 3.905, 0.001)
        
        distortion = st.selectbox("Lowest Distortion", 
                                ["cubic", "tetragonal", "orthorhombic", "rhombohedral"])
    
    if st.button("🚀 Predict Stability", use_container_width=True):
        with st.spinner("Predicting stability..."):
            # Prepare features
            features = np.array([
                valence_a, valence_b, radius_a, radius_b,
                formation_energy, band_gap, magnetic_moment,
                a, b, c, 90.0, 90.0, 90.0  # Assuming 90° angles
            ])
            
            # Predict
            try:
                stability = st.session_state.predictor.predict_stability(features)
                display_prediction_result(stability, features, {
                    'A': a_element, 'B': b_element, 'Distortion': distortion
                })
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                stability = 0.8  # Demo value
                display_prediction_result(stability, features, {
                    'A': a_element, 'B': b_element, 'Distortion': distortion
                })

def csv_upload_section():
    st.markdown('<h2 class="sub-header">📁 CSV File Upload</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV file with compound data", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} compounds")
            
            # Show preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            
            # Check required columns
            required_cols = ['Valence A', 'Valence B', 'Radius A [ang]', 'Radius B [ang]',
                           'Formation energy [eV/atom]', 'Band gap [eV]', 'Magnetic moment [mu_B]',
                           'a', 'b', 'c']
            
            if all(col in df.columns for col in required_cols):
                if st.button("🔮 Predict All Compounds", use_container_width=True):
                    predict_batch_compounds(df)
            else:
                st.error(f"Missing required columns. Needed: {required_cols}")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file with compound data to get started.")

def sample_data_section():
    st.markdown('<h2 class="sub-header">🧪 Sample Data Demonstration</h2>', unsafe_allow_html=True)
    
    df = create_sample_data()
    st.dataframe(df)
    
    if st.button("🎯 Predict Sample Compounds", use_container_width=True):
        predict_batch_compounds(df)

def predict_batch_compounds(df):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    predictions = []
    features_list = []
    
    for i, row in df.iterrows():
        status_text.text(f"Processing compound {i+1}/{len(df)}...")
        progress_bar.progress((i + 1) / len(df))
        
        try:
            # Prepare features (assuming standard columns)
            features = np.array([
                row['Valence A'], row['Valence B'],
                row['Radius A [ang]'], row['Radius B [ang]'],
                row['Formation energy [eV/atom]'], row['Band gap [eV]'],
                row['Magnetic moment [mu_B]'],
                row.get('a', 3.9), row.get('b', 3.9), row.get('c', 3.9),
                90.0, 90.0, 90.0  # Default angles
            ])
            
            stability = st.session_state.predictor.predict_stability(features)
            predictions.append(stability)
            features_list.append(features)
            
        except Exception as e:
            st.warning(f"Error processing row {i}: {str(e)}")
            predictions.append(np.nan)
    
    # Add predictions to dataframe
    df['Predicted Stability [eV/atom]'] = predictions
    
    # Display results
    st.markdown("### 📈 Prediction Results")
    st.dataframe(df)
    
    # Visualizations
    create_batch_visualizations(df, predictions)

def create_batch_visualizations(df, predictions):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Trends", "🔍 Correlations", "⏰ Time Series"])
    
    with tab1:
        fig = px.histogram(df, x='Predicted Stability [eV/atom]', 
                          title='Distribution of Predicted Stability',
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'A' in df.columns and 'B' in df.columns:
            df['Compound'] = df['A'] + df['B'] + 'O₃'
            fig = px.bar(df, x='Compound', y='Predicted Stability [eV/atom]',
                        title='Stability by Compound',
                        color='Predicted Stability [eV/atom]',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title='Feature Correlation Matrix',
                          color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if len(predictions) > 10:  # Only show time series for sufficient data
            create_time_series_analysis(predictions)

def create_time_series_analysis(predictions):
    # Create synthetic time index for demonstration
    dates = pd.date_range(start='2023-01-01', periods=len(predictions), freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'stability': predictions,
        'rolling_mean': pd.Series(predictions).rolling(window=7).mean()
    })
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Stability Trend', 'Seasonal Decomposition'))
    
    # Trend plot
    fig.add_trace(
        go.Scatter(x=ts_data['date'], y=ts_data['stability'], 
                  name='Stability', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ts_data['date'], y=ts_data['rolling_mean'], 
                  name='7-day Moving Avg', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    # Seasonal decomposition (simplified)
    from statsmodels.tsa.seasonal import seasonal_decompose
    try:
        result = seasonal_decompose(ts_data['stability'].dropna(), period=7, model='additive')
        fig.add_trace(
            go.Scatter(x=ts_data['date'], y=result.trend, name='Trend', line=dict(color='#2ca02c')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=ts_data['date'], y=result.seasonal, name='Seasonal', line=dict(color='#d62728')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=ts_data['date'], y=result.resid, name='Residual', line=dict(color='#9467bd')),
            row=2, col=1
        )
    except:
        st.warning("Insufficient data for full time series decomposition")
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def display_prediction_result(stability, features, metadata):
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    
    # Determine stability category
    if stability < 0.1:
        stability_class = "stability-good"
        stability_text = "Excellent Stability"
    elif stability < 0.5:
        stability_class = "stability-medium"
        stability_text = "Good Stability"
    else:
        stability_class = "stability-poor"
        stability_text = "Poor Stability"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Prediction Result")
        st.markdown(f"**Compound:** {metadata['A']}{metadata['B']}O₃")
        st.markdown(f"**Distortion:** {metadata['Distortion']}")
        st.markdown(f"**Predicted Stability:** <span class='{stability_class}'>{stability:.4f} eV/atom</span>", 
                   unsafe_allow_html=True)
        st.markdown(f"**Category:** <span class='{stability_class}'>{stability_text}</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        # Create radar chart for feature visualization
        feature_names = ['Valence A', 'Valence B', 'Radius A', 'Radius B', 
                        'Form Energy', 'Band Gap', 'Mag Moment']
        feature_values = features[:7]  # First 7 features
        
        # Normalize for radar chart
        max_vals = [4, 6, 2.0, 1.5, 1.0, 6.0, 5.0]
        normalized_values = [v/max_vals[i] for i, v in enumerate(feature_values)]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],  # Close the circle
            theta=feature_names + [feature_names[0]],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line=dict(color='rgb(31, 119, 180)')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Feature Radar Chart",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional visualizations
    create_individual_visualizations(stability, features)

def create_individual_visualizations(stability, features):
    col1, col2 = st.columns(2)
    
    with col1:
        # Stability gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = stability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Stability Score"},
            gauge = {
                'axis': {'range': [0, 2]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.1], 'color': "lightgreen"},
                    {'range': [0.1, 0.5], 'color': "yellow"},
                    {'range': [0.5, 2], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': stability
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance bar chart
        feature_names = ['Val A', 'Val B', 'Rad A', 'Rad B', 'Form E', 'Band G', 'Mag M',
                        'a', 'b', 'c', 'α', 'β', 'γ']
        importance = np.abs(features) / np.max(np.abs(features))
        
        fig = px.bar(x=feature_names, y=importance, 
                    title="Feature Importance (Absolute Values)",
                    color=importance,
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, xaxis_title="Features", yaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()