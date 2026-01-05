# Quantum Error Mitigation using Machine Learning

A comprehensive machine learning framework for mitigating quantum computing errors using ensemble learning and noise-aware neural networks.

## 📋 Project Overview

This project implements advanced quantum error mitigation (QEM) techniques using two complementary approaches:

1. **Stacked Ensemble Model**: Combines XGBoost, Neural Networks (with LSTM), and Random Forest for robust error correction
2. **Noise-Aware QEM Model**: Two-stage neural architecture that learns noise descriptors and applies conditional mitigation

## 🎯 Key Features

- **Multi-Model Ensemble**: GPU-accelerated stacking of XGBoost, PyTorch Neural Networks, and Random Forest
- **LSTM-Enhanced Neural Network**: Advanced temporal pattern recognition for quantum error mitigation
- **Noise-Aware Architecture**: Dedicated noise descriptor network for adaptive mitigation
- **Comprehensive Dataset Generation**: Support for variational circuits, QAOA, and random quantum circuits
- **Multiple Noise Models**: Depolarizing, amplitude damping, and readout error support
- **Extensive Benchmarking**: Evaluation across circuit depths, qubit counts, and error rates

## 🚀 Performance Highlights

- **Error Reduction**: Up to 85%+ improvement over baseline noisy measurements
- **Mitigation Factor (R)**: Achieves R > 5 for many circuit configurations
- **Scalability**: Tested on 4-12 qubit systems with depths up to 10
- **GPU Acceleration**: Leverages CUDA for fast training and inference

## 📁 Repository Contents

```
quantum-error-mitigation/
│
├── notebooks/
│   ├── 01_data_generation.ipynb            #data set generation code
│   ├── 02_data_exploration.ipynb           # EDA and statistical analysis
|   ├── 03_stacked_ensemble_training.ipynb  # XGBoost + NN + RF ensemble
│   └── 04_noise_aware_qem.ipynb            # Two-stage noise-aware model
│
├── data/
│   └── sample_data                         # Sample datasets (if included)
│
├── team7 presentation.pdf                  #the ppt of the presentaion
├── QEM_AQC.pdf                             #the report
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── LICENSE                                 # MIT License
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)
- 8GB+ RAM recommended

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/quantum-error-mitigation.git
cd quantum-error-mitigation
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv qem_env
source qem_env/bin/activate  # On Windows: qem_env\Scripts\activate

# OR using conda
conda create -n qem_env python=3.9
conda activate qem_env
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For GPU support**, ensure CUDA is installed and run:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Verify installation**
```python
import torch
import qiskit
import xgboost

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Qiskit version: {qiskit.__version__}")
print(f"XGBoost version: {xgboost.__version__}")
```

## 📊 Quick Start

### Running the Notebooks

**Option 1: Jupyter Notebook**
```bash
jupyter notebook
# Navigate to notebooks/ folder and open any notebook
```

**Option 2: Jupyter Lab**
```bash
jupyter lab
```

**Option 3: Google Colab**
- Upload notebooks to Google Drive
- Open with Google Colab
- Mount drive and update file paths as needed

### Notebook Execution Order

1. **01_data_exploration.ipynb**
   - Load and inspect quantum error mitigation dataset
   - Perform exploratory data analysis
   - Visualize error distributions and correlations
   - Analyze noise type impacts

2. **02_stacked_ensemble_training.ipynb**
   - Feature engineering and preprocessing
   - Train XGBoost (GPU-accelerated)
   - Train Neural Network with LSTM (PyTorch)
   - Train Random Forest
   - Combine models using Ridge meta-learner
   - Evaluate and visualize results
   - **Expected Performance**: ~82% error reduction

3. **03_noise_aware_qem.ipynb**
   - Generate quantum circuits and datasets
   - Implement two-stage noise-aware architecture
   - Train noise descriptor network
   - Train conditional mitigation network
   - Benchmark on various circuit types
   - **Expected Mitigation Factor**: R > 5

## 🔬 Model Architectures

### Stacked Ensemble (Notebook 2)

```
Base Models:
├── XGBoost (GPU-accelerated)
│   ├── n_estimators: 500
│   ├── max_depth: 6
│   └── tree_method: 'gpu_hist'
│
├── Neural Network (PyTorch with LSTM)
│   ├── LSTM(input → 128)
│   ├── BatchNorm1d(128)
│   ├── Dense(128 → 64 → 32 → 16 → 1)
│   └── Dropout(0.2)
│
└── Random Forest
    ├── n_estimators: 300
    └── max_depth: 15

Meta-Learner: Ridge Regression (α=1.0)
```

### Noise-Aware QEM (Notebook 3)

```
Stage 1: Noise Descriptor Network g(x_n)
  Input → 64 → 32 → 16 (latent z)
  - BatchNorm + ReLU + Dropout

Stage 2: Conditional Mitigation Network h(x_n, z)
  [Input + Latent] → 64 → 32 → 1
  - Combines noisy input with learned noise features
```

## 📈 Results

### Performance Metrics (Stacked Ensemble)

| Model | MAE | RMSE | R² | Improvement | Training Time |
|-------|-----|------|-----|-------------|---------------|
| Baseline (No Mitigation) | 0.423 | 0.519 | -0.12 | 0% | - |
| XGBoost | 0.087 | 0.124 | 0.92 | 79.4% | 45s |
| Neural Network (LSTM) | 0.082 | 0.118 | 0.93 | 80.6% | 120s |
| Random Forest | 0.091 | 0.131 | 0.91 | 78.5% | 38s |
| **Stacked Ensemble** | **0.076** | **0.109** | **0.94** | **82.0%** | 203s |

### Mitigation Factor by Circuit Depth

| Depth | Baseline Error | Model Error | Mitigation Factor (R) |
|-------|---------------|-------------|----------------------|
| 2 | 0.234 | 0.042 | 5.57 |
| 4 | 0.389 | 0.068 | 5.72 |
| 6 | 0.512 | 0.091 | 5.63 |
| 8 | 0.634 | 0.119 | 5.33 |
| 10 | 0.748 | 0.152 | 4.92 |

### Key Findings

- **Stacked ensemble outperforms individual models** by 2-4% in error reduction
- **LSTM integration** improves neural network performance by capturing temporal dependencies
- **Noise-aware model** achieves consistent mitigation factor R > 5 across depths
- **GPU acceleration** reduces training time by 10-15x for large datasets
- **Feature importance analysis** shows x_noisy, error_rate, and depth are most critical

## 📝 Dataset Information

The project uses quantum error mitigation datasets with the following features:

### Input Features
- `x_noisy`: Noisy expectation value from quantum circuit
- `num_qubits`: Number of qubits (4, 8, 12)
- `depth`: Circuit depth (1-10)
- `error_rate`: Noise error rate (0.001, 0.01, 0.1)
- `noise_type`: Type of noise (depolarizing, amplitude_damping, readout)
- `entanglement`: Entanglement pattern (linear, full, pairwise)
- `observable_name`: Observable being measured

### Target Variable
- `x_ideal`: Ideal (noise-free) expectation value

### Dataset Statistics
- **Training samples**: ~70% of total
- **Validation samples**: ~15% of total
- **Test samples**: ~15% of total
- **Total configurations**: 81+ unique circuit-noise combinations

## 🔧 Customization

### Modifying Hyperparameters

**In Notebook 2 (Stacked Ensemble):**
```python
# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=500,      # Adjust number of trees
    learning_rate=0.05,    # Adjust learning rate
    max_depth=6,           # Adjust tree depth
    tree_method='gpu_hist' # Use 'hist' for CPU
)

# Neural Network
epochs = 150               # Adjust training epochs
batch_size = 256          # Adjust batch size
latent_dim = 16           # Adjust LSTM hidden size
```

**In Notebook 3 (Noise-Aware):**
```python
# Model configuration
model = NoiseAwareQEMModel(
    input_dim=input_features,
    latent_dim=16          # Adjust latent space size
)

# Training configuration
num_epochs = 100           # Adjust epochs
learning_rate = 0.001      # Adjust learning rate
batch_size = 32           # Adjust batch size
```

## 🧪 Testing and Validation

Each notebook includes comprehensive testing:

### Notebook 1 (EDA)
- Data quality checks
- Missing value analysis
- Statistical summaries
- Correlation analysis

### Notebook 2 (Ensemble)
- Individual model evaluation
- Cross-validation (K-Fold)
- Residual analysis
- Feature importance validation

### Notebook 3 (Noise-Aware)
- Benchmarking on unseen circuits
- Generalization to different qubit counts
- Robustness across noise types
- Depth scaling analysis

## 🐛 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check CUDA
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory errors:**
```python
# Reduce batch size
batch_size = 64  # or 32

# Or use CPU
device = 'cpu'
```

**Import errors:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

**Qiskit compatibility:**
```bash
# Update Qiskit
pip install --upgrade qiskit qiskit-aer
```

## 📚 Dependencies

Key packages used:
- **Qiskit 1.0+**: Quantum circuit simulation
- **PyTorch 2.0+**: Neural network implementation
- **XGBoost 2.0+**: Gradient boosting
- **Scikit-learn 1.3+**: Random Forest and preprocessing
- **Pandas 2.0+**: Data manipulation
- **NumPy 1.24+**: Numerical computing
- **Matplotlib/Seaborn**: Visualization

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{qem_ml_2024,
  title = {Quantum Error Mitigation using Machine Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/quantum-error-mitigation}
}
```

## 🙏 Acknowledgments

- Built with Qiskit for quantum circuit simulation
- Uses PyTorch, XGBoost, and Scikit-learn for machine learning
- Inspired by recent advances in quantum error mitigation research

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project Link**: https://github.com/yourusername/quantum-error-mitigation

---

**Version**: 1.0.0 | **Last Updated**: January 2026 | **Status**: Active Development