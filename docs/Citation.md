# Citations and References

## Code Attribution

### Original Work (Written by Pranav)

The following components were designed and implemented from scratch for this project:

1. **Experiment Framework**
   - `SpeedRunner` class for automated experiment execution
   - Checkpoint recovery system with corruption handling
   - Memory-optimized training pipeline
   - Comprehensive result tracking and comparison system

2. **Speed Optimizations**
   - Frame skip wrapper implementation
   - Optimized replay buffer sizing (50k → 30k)
   - Reduced warmup period (10k → 6k)
   - GPU memory management and garbage collection

3. **Boltzmann Exploration Implementation**
   - `DQNAgentBoltzmann` class with temperature-based action selection
   - Temperature decay schedule design
   - Comparative analysis framework (ε-greedy vs Boltzmann)

4. **Visualization and Analysis**
   - Multi-panel comparison plots
   - Learning curve analysis tools
   - Statistical significance testing
   - Automated report generation

5. **Documentation**
   - Comprehensive README
   - Methodology documentation
   - Results analysis
   - This attribution document

### Adapted from Existing Work

The following components were adapted from published research and standard implementations:

#### 1. DQN Architecture (Mnih et al., 2015)

**Source:** Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

**Original Architecture:**
```python
Conv1: 32 filters, 8×8 kernel, stride 4, ReLU
Conv2: 64 filters, 4×4 kernel, stride 2, ReLU
Conv3: 64 filters, 3×3 kernel, stride 1, ReLU
FC1: 512 units, ReLU
FC2: num_actions units (Q-values)
```

**Adaptations Made:**
- Implemented in PyTorch (original used Torch7)
- Added gradient clipping for stability
- Optimized for modern GPU (Tesla P100)

**License:** Nature paper (research/educational use)

#### 2. Experience Replay (Mnih et al., 2013)

**Source:** Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013). "Playing Atari with Deep Reinforcement Learning." *ArXiv:1312.5602*.

**Original Concept:**
- Store transitions (s, a, r, s', done) in buffer
- Sample random minibatches for training
- Break temporal correlation in sequential data

**Adaptations Made:**
- Reduced buffer size for efficiency (50k → 30k)
- Optimized sampling mechanism
- Added checkpoint save/load functionality

#### 3. Frame Preprocessing (OpenAI Baselines)

**Source:** OpenAI Baselines - [github.com/openai/baselines](https://github.com/openai/baselines)

**Standard Preprocessing:**
- RGB to grayscale conversion
- Resize to 84×84
- Normalization to [0, 1]
- Frame stacking (4 frames)

**Adaptations Made:**
- Implemented using PIL instead of OpenCV
- Optimized for batch processing
- Added frame skip wrapper

**License:** MIT License (OpenAI)

#### 4. Frame Skip Technique (Bellemare et al., 2013)

**Source:** Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). "The Arcade Learning Environment: An Evaluation Platform for General Agents." *Journal of Artificial Intelligence Research*, 47, 253-279.

**Original Concept:**
- Repeat action for k frames (k=4 standard)
- Return max of last 2 frames (handle flickering)
- Reduces computation by 75%

**Adaptations Made:**
- Implemented as Gymnasium wrapper
- Configurable skip parameter
- Integrated with preprocessing pipeline

---

## Libraries and Frameworks

### Core Dependencies

#### PyTorch 2.0+
- **Purpose:** Neural network implementation and training
- **Website:** [pytorch.org](https://pytorch.org)
- **License:** BSD 3-Clause
- **Citation:**
```bibtex
  @inproceedings{paszke2019pytorch,
    title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
    author={Paszke, Adam and Gross, Sam and Massa, Francisco and others},
    booktitle={NeurIPS},
    year={2019}
  }
```

#### Gymnasium 1.0+
- **Purpose:** RL environment interface (successor to OpenAI Gym)
- **Website:** [gymnasium.farama.org](https://gymnasium.farama.org)
- **License:** MIT License
- **Citation:**
```bibtex
  @software{towers_gymnasium_2024,
    title={Gymnasium},
    author={Towers, Mark and others},
    year={2024},
    publisher={Farama Foundation},
    url={https://github.com/Farama-Foundation/Gymnasium}
  }
```

#### Arcade Learning Environment (ALE)
- **Purpose:** Atari 2600 game emulation
- **Website:** [github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
- **License:** GPL v2
- **Citation:**
```bibtex
  @article{bellemare2013arcade,
    title={The Arcade Learning Environment: An Evaluation Platform for General Agents},
    author={Bellemare, Marc G and Naddaf, Yavar and Veness, Joel and Bowling, Michael},
    journal={Journal of Artificial Intelligence Research},
    volume={47},
    pages={253--279},
    year={2013}
  }
```

### Supporting Libraries

- **NumPy 1.24+:** Numerical computations (BSD License)
- **Matplotlib 3.7+:** Visualization (PSF License)
- **Pandas 2.0+:** Data analysis (BSD License)
- **Pillow 10.0+:** Image processing (HPND License)

---

## Research References

### Foundational Papers

#### 1. Deep Q-Network (DQN)

**Main Paper:**
```bibtex
@article{mnih2015human,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
  journal={Nature},
  volume={518},
  number={7540},
  pages={529--533},
  year={2015},
  publisher={Nature Publishing Group}
}
```

**ArXiv Preprint:**
```bibtex
@article{mnih2013playing,
  title={Playing Atari with Deep Reinforcement Learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  journal={arXiv preprint arXiv:1312.5602},
  year={2013}
}
```

#### 2. Reinforcement Learning Textbook
```bibtex
@book{sutton2018reinforcement,
  title={Reinforcement Learning: An Introduction},
  author={Sutton, Richard S and Barto, Andrew G},
  year={2018},
  publisher={MIT Press},
  edition={2nd}
}
```

**Relevant Chapters:**
- Chapter 6: Temporal-Difference Learning (Q-learning algorithm)
- Chapter 9: On-policy Prediction with Approximation
- Chapter 10: On-policy Control with Approximation

#### 3. Exploration Strategies

**Boltzmann Exploration:**
```bibtex
@article{cesa2017boltzmann,
  title={Boltzmann Exploration Done Right},
  author={Cesa-Bianchi, Nicolo and Gentile, Claudio and Lugosi, Gabor and Neu, Gergely},
  journal={NeurIPS},
  pages={6284--6293},
  year={2017}
}
```

**Epsilon-Greedy Analysis:**
```bibtex
@inproceedings{auer2002finite,
  title={Finite-time analysis of the multiarmed bandit problem},
  author={Auer, Peter and Cesa-Bianchi, Nicolo and Fischer, Paul},
  booktitle={Machine Learning},
  volume={47},
  number={2-3},
  pages={235--256},
  year={2002}
}
```

#### 4. Experience Replay
```bibtex
@article{lin1992self,
  title={Self-improving reactive agents based on reinforcement learning, planning and teaching},
  author={Lin, Long-Ji},
  journal={Machine Learning},
  volume={8},
  number={3-4},
  pages={293--321},
  year={1992}
}
```

#### 5. Target Networks
```bibtex
@article{lillicrap2015continuous,
  title={Continuous control with deep reinforcement learning},
  author={Lillicrap, Timothy P and Hunt, Jonathan J and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  journal={arXiv preprint arXiv:1509.02971},
  year={2015}
}
```

### Related DQN Improvements

#### Double DQN
```bibtex
@inproceedings{van2016deep,
  title={Deep Reinforcement Learning with Double Q-learning},
  author={Van Hasselt, Hado and Guez, Arthur and Silver, David},
  booktitle={AAAI},
  volume={30},
  number={1},
  year={2016}
}
```

#### Dueling DQN
```bibtex
@inproceedings{wang2016dueling,
  title={Dueling Network Architectures for Deep Reinforcement Learning},
  author={Wang, Ziyu and Schaul, Tom and Hessel, Matteo and Van Hasselt, Hado and Lanctot, Marc and De Freitas, Nando},
  booktitle={ICML},
  pages={1995--2003},
  year={2016}
}
```

#### Prioritized Experience Replay
```bibtex
@inproceedings{schaul2015prioritized,
  title={Prioritized Experience Replay},
  author={Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
  booktitle={ICLR},
  year={2016}
}
```

#### Rainbow DQN (Combination of Improvements)
```bibtex
@inproceedings{hessel2018rainbow,
  title={Rainbow: Combining Improvements in Deep Reinforcement Learning},
  author={Hessel, Matteo and Modayil, Joseph and Van Hasselt, Hado and Schaul, Tom and Ostrovski, Georg and Dabney, Will and Horgan, Dan and Piot, Bilal and Azar, Mohammad and Silver, David},
  booktitle={AAAI},
  volume={32},
  number={1},
  year={2018}
}
```

---

## Educational Resources

### Online Courses

1. **Deep Reinforcement Learning (CS 285) - UC Berkeley**
   - Instructor: Sergey Levine
   - URL: [rail.eecs.berkeley.edu/deeprlcourse/](http://rail.eecs.berkeley.edu/deeprlcourse/)

2. **Reinforcement Learning - David Silver (DeepMind)**
   - Instructor: David Silver
   - URL: [youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

3. **MIT 6.S091: Introduction to Deep Reinforcement Learning**
   - Instructor: Lex Fridman
   - URL: [deeplearning.mit.edu](https://deeplearning.mit.edu)

### Tutorials and Blog Posts

1. **Deep RL Course - Hugging Face**
   - URL: [huggingface.co/deep-rl-course](https://huggingface.co/deep-rl-course)

2. **OpenAI Spinning Up in Deep RL**
   - URL: [spinningup.openai.com](https://spinningup.openai.com)

3. **Lilian Weng's Blog - Deep Reinforcement Learning**
   - URL: [lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

---

## Platform and Infrastructure

### Kaggle
- **Platform:** Cloud-based Jupyter notebook environment
- **GPU:** Tesla P100 (16GB VRAM) provided free of charge
- **Website:** [kaggle.com](https://www.kaggle.com)
- **License:** Kaggle Terms of Service
- **Usage:** Training and experimentation platform

### Northeastern University
- **Institution:** Northeastern University, Boston, MA
- **Program:** MS Information Systems
- **Course:** Advanced Topics in Machine Learning / Reinforcement Learning
- **Instructor:** [Course Instructor Name]
- **Semester:** Fall 2025

---

## Acknowledgments

### Technical Guidance
- **Anthropic Claude:** AI assistant for debugging, optimization suggestions, and documentation
- **Kaggle Community:** Discussion forums and example notebooks
- **PyTorch Community:** Documentation and tutorials

### Inspirational Projects
- **OpenAI Baselines:** Reference implementations of RL algorithms
- **Stable Baselines3:** High-quality RL implementations
- **RLlib (Ray):** Scalable RL library

### Academic Support
- **Northeastern University Faculty:** Course instruction and project guidance
- **Peer Reviewers:** Classmates who provided feedback on methodology

---

## Data and Code Availability

### GitHub Repository
- **URL:** [github.com/YOUR_USERNAME/DQN-DonkeyKong-Atari](https://github.com/YOUR_USERNAME/DQN-DonkeyKong-Atari)
- **License:** MIT License (see LICENSE file)
- **Contents:**
  - Complete source code
  - Jupyter notebook
  - Experiment results (plots, data files)
  - Documentation (README, methodology, results)
  - Trained model checkpoints (available on request)

### Kaggle Notebook
- **URL:** [kaggle.com/YOUR_USERNAME/dqn-donkeykong-experiments](https://www.kaggle.com/YOUR_USERNAME/dqn-donkeykong-experiments)
- **Public Access:** Yes
- **Includes:** Interactive notebook with outputs

### Data Files
All experimental results available in the repository:
- CSV files: Episode-by-episode rewards and steps
- JSON file: Experiment summary and configuration
- PNG files: Visualizations and plots
- TXT files: Detailed reports

---

## License

This project is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2025 Pranav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Rationale for MIT License
- **Permissive:** Allows commercial and non-commercial use
- **Attribution:** Requires credit to original author
- **Industry Standard:** Widely used in open-source ML projects
- **Education-Friendly:** No restrictions on academic use

---

## Ethical Considerations

### Reproducibility
- All random seeds fixed (SEED=42)
- Hyperparameters explicitly documented
- Code publicly available
- Results independently verifiable

### Transparency
- Clear attribution of adapted code
- Honest reporting of limitations
- No selective result reporting (all 5 experiments reported)
- Negative results acknowledged where applicable

### Responsible AI
- Research application only (no deployment)
- No harmful applications
- Educational purpose emphasized
- Open-source contribution to community

---

## Contact and Feedback

### Author
**Name:** Pranav  
**Institution:** Northeastern University  
**Program:** MS Information Systems  
**Email:** [your.email@northeastern.edu]  
**LinkedIn:** [linkedin.com/in/your-profile]  
**GitHub:** [github.com/YOUR_USERNAME]

### How to Cite This Work

If you use this code, methodology, or results in your research, please cite:
```bibtex
@misc{pranav2025dqn_donkeykong,
  title={Deep Q-Learning for Atari DonkeyKong: Hyperparameter Optimization and Policy Exploration},
  author={Pranav},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/YOUR_USERNAME/DQN-DonkeyKong-Atari}},
  note={MS Information Systems Project, Northeastern University}
}
```

---

## Version History

- **v1.0.0** (November 2025): Initial release
  - Complete DQN implementation
  - 5 experimental configurations
  - Comprehensive documentation
  - All results and visualizations

---

**Last Updated:** November 7, 2025  
**Document Version:** 1.0  
**Author:** Pranav, Northeastern University
