# AdaDRO
AdaDRO: Adaptive Robust Classification Driven by Optimal Transport

 "Bootstrap Your Uncertainty: Adaptive Robust Classification Driven by Optimal-Transport" (submitted to NeurIPS 2025).

## Start

### Installation

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy
```

### Basic Usage

```bash
python main.py --dataset cifar10 --epochs 100 --batch-size 128

python main.py --dataset colored_mnist --epochs 50 --batch-size 256

python main.py --dataset cifar100 --epochs 200 --batch-size 64
```

## 📁 Project Structure

```
adadro_project/
├── main.py                   
├── config/                    
├── models/                  
│   ├── adadro_model.py        
│   ├── moco.py               
│   └── backbone.py           
├── losses/                    
│   ├── adadro_loss.py        
│   └── optimal_transport.py 
├── data/                     
├── utils/                   
│   ├── filtering.py         
│   ├── mlmc.py              
│   └── metrics.py          
└── training/                 
```

##  Key Features

### Two-Stage Training

1. **Semantic Calibration**: Learn semantic transport costs via inverse OT
   - Feature space IOT: MoCo InfoNCE loss
   - Label space IOT: Cross-entropy loss
2. **Adaptive DRO**: Robust optimization with dynamic uncertainty sets

### Core Components

- **Adaptive Filtering**: OT-driven reference distribution refinement
- **Semantic Transport Costs**: Cosine similarity-based feature/label costs
- **Worst-case Distribution**: Sinkhorn DRO with evolving uncertainty sets
- **MLMC Gradient Estimation**: Efficient gradient computation

##  Configuration

```bash
python main.py \
    --dataset cifar10 \           
    --arch resnet18 \           
    --epochs 200 \              
    --batch-size 256 \            
    --lr 0.01 \                  
    --lambda-reg 1.0 \           
    --kappa 1.0 \                
    --device cuda \              
    --experiment-name my_exp      
```

##  Algorithm Overview

### Core Mathematical Formulation

**Semantic Calibration (IOT Problems):**

```
min_θ KL(γ̄ˣ | γᵗˣ), s.t. γᵗˣ = argmin E[Cᵗˣ(xᵢ,x'ⱼ)] + εH(γ)
min_θ KL(γ̄ | γθ), s.t. γθ = argmin E[Cθ(qᵢ,k)] + εH(γ)
```

**Adaptive Filtering:**

```
ν̂(p) = ν(p)·𝟙[∃q: γ̄ˣ(p,q)>0 ∧ γᵗˣ(p,q)≥τ(q)] / χ
```

**Worst-case Distribution:**

```
Qᵏ(q) = Eₚ[exp((ℓ(q)-λC(p,q))/(λε)) / Z(p)] · ν̂(q)
```

### Theoretical Guarantees

- **Convergence**: O(ε⁻⁴log²(1/ε)) sample complexity
- **Adaptivity**: Dynamic uncertainty set evolution
- **Robustness**: Performance guarantees under distribution shift

