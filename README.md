# 🎮 Deep Q-Learning for Atari DonkeyKong

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep Q-Network implementation for Atari DonkeyKong-v5 with comprehensive hyperparameter optimization and policy exploration experiments.

**Author:** Pranav  
**Institution:** Northeastern University  
**Course:** MS Information Systems  
**Date:** November 2025

---

## 📊 **Results Summary**

| Experiment | Avg Reward | vs Baseline | Improvement |
|------------|-----------|-------------|-------------|
| **Baseline** | 122 | --- | --- |
| Higher Learning Rate | 810 | +688 | **+564%** |
| Lower Gamma | 1,083 | +961 | **+788%** |
| Lower Min Epsilon | 1,146 | +1,024 | **+839%** |
| Faster Decay | 1,053 | +931 | **+763%** |
| **Boltzmann Softmax** | **3,567** | **+3,445** | **+2,824%** 🏆 |

**Key Finding:** Boltzmann/Softmax exploration dramatically outperformed epsilon-greedy by using probabilistic action selection based on Q-values.

---

## 🎯 **Project Overview**

This project implements a Deep Q-Network (DQN) agent to play Atari DonkeyKong, exploring:

- **Bellman Equation Parameters:** Learning rate (α) and discount factor (γ)
- **Exploration Strategies:** Epsilon-greedy variations and Boltzmann exploration
- **Policy Comparison:** Value-based vs probabilistic action selection

### **Technical Highlights**

✅ Neural network with ~1.7M parameters  
✅ Frame skip optimization (4x speedup)  
✅ Experience replay buffer (30k transitions)  
✅ Target network stabilization  
✅ Comprehensive experiment framework  

---

## 📁 **Repository Structure**
```
├── notebooks/          # Jupyter notebooks
├── results/
│   ├── plots/         # Visualizations
│   ├── data/          # CSV and JSON results
│   └── reports/       # Text reports
├── docs/              # Documentation
└── assets/            # Demo video and images
```

---

## 🚀 **Quick Start**

### **Requirements**
```bash
pip install -r requirements.txt
```

### **Run Training**
```bash
# Open Kaggle notebook or run locally
jupyter notebook notebooks/DQN_Training.ipynb
```

### **View Results**

All results available in `results/` directory:
- **Plots:** `results/plots/FINAL_RESULTS.png`
- **Data:** `results/data/summary.json`
- **Report:** `results/reports/ASSIGNMENT_REPORT.txt`

---

## 📈 **Visualizations**

### Main Results
![Final Results](results/plots/FINAL_RESULTS.png)

### Boltzmann Analysis
![Boltzmann Analysis](results/plots/Boltzmann_Analysis.png)

### Checkpoint Progression
![Checkpoint Analysis](results/plots/Checkpoint_Analysis_1000.png)

---

## 🧠 **Methodology**

### **DQN Architecture**
- **Input:** 4 stacked grayscale frames (84×84 pixels)
- **Conv Layers:** 3 layers (32→64→64 filters)
- **FC Layers:** 512 units → 18 actions
- **Optimizer:** Adam (lr=0.00025)

### **Training Configuration**
- **Episodes:** 1,500 (baseline) + 500 (experiments)
- **Replay Buffer:** 30,000 transitions
- **Target Update:** Every 1,000 steps
- **Frame Skip:** 4x (standard Atari optimization)

### **Experiments**

1. **Bellman Parameters**
   - Learning Rate: 0.00025 → 0.0005
   - Discount Factor: 0.99 → 0.95

2. **Exploration Parameters**
   - Min Epsilon: 0.1 → 0.01
   - Decay Rate: 0.995 → 0.99

3. **Policy Exploration**
   - Alternative: Boltzmann/Softmax with temperature parameter
   - Temperature decay: 1.0 → 0.1

---

## 📊 **Key Findings**

### **1. Learning Rate Impact**
Higher learning rate (2x) improved convergence speed by 564%, enabling faster policy optimization.

### **2. Discount Factor**
Lower gamma (0.95) improved performance by 788%, suggesting DonkeyKong benefits from immediate reward focus.

### **3. Exploration Strategy**
Boltzmann exploration achieved **2,824% improvement**, demonstrating that probabilistic action selection based on Q-values vastly outperforms random exploration.

### **4. Convergence**
All experiments converged within 300-400 episodes, with Boltzmann showing the smoothest learning curve.

---

## 📝 **Theoretical Analysis**

### **Q-Learning Classification**
DQN is **value-based** because:
- Learns Q-values Q(s,a) explicitly
- Derives policy via argmax (greedy selection)
- Updates values, not policy parameters directly

### **Bellman Equation**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- α (alpha): Learning rate
- γ (gamma): Discount factor
- r: Immediate reward
- max_a' Q(s',a'): Best future value

---

## 🎬 **Demo Video**

[Link to demo video] *(Upload to YouTube/Loom and add link)*

---

## 📚 **References**

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
2. Mnih, V., et al. (2013). "Playing Atari with Deep Reinforcement Learning." *ArXiv:1312.5602*.
3. OpenAI Gymnasium: https://gymnasium.farama.org/
4. Arcade Learning Environment: https://github.com/mgbellemare/Arcade-Learning-Environment

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Architecture:** Based on Mnih et al. (2015) DQN paper
- **Framework:** PyTorch, Gymnasium, ALE
- **Platform:** Kaggle (Tesla P100 GPU)
- **Course:** MS Information Systems, Northeastern University

---

## 👤 **Author**

**Pranav**  
MS Information Systems Student  
Northeastern University  
📧 kharat.p@northeastern.edu


---





**Built with ❤️ for reinforcement learning research**
