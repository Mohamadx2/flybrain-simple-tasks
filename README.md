# FlyBrain Simple Tasks

Train simple readouts on a **fruit-fly (MaleCNS-inspired) mapped neural circuit** for basic tasks.

This is an educational starter project inspired by the [awesome-fly](https://github.com/cobanov/awesome-fly) collection and the [fly-connectome-template](https://github.com/cobanov/fly-connectome-template).

> **Important**: This is **not** a full biological brain simulation.  
> We use a small, fixed circuit structure inspired by real MaleCNS connectivity patterns, then train only a lightweight readout (or a few gains) for simple tasks.

## Available Tasks

### 1. CartPole-lite (classic control)
Train the circuit to balance a pole.

### 2. Word Association (NEW) — Treat & Punishment
Train the circuit to associate simple words with positive or negative outcomes, similar in spirit to how apes were trained with rewards and corrections.

| Word     | True meaning | Desired action |
|----------|--------------|----------------|
| banana   | good         | approach       |
| apple    | good         | approach       |
| safe     | good         | approach       |
| good     | good         | approach       |
| water    | good         | approach       |
| danger   | bad          | avoid          |
| bad      | bad          | avoid          |
| fire     | bad          | avoid          |

- Correct association → **+1 (treat)**
- Wrong association → **-1 (punishment)**

## Project Structure

```
flybrain-simple-tasks/
├── circuit/
│   └── male_cns_subset.py      # Fixed fly-inspired circuit
├── tasks/
│   ├── cartpole_lite.py
│   └── word_association.py     # NEW – treat / punishment task
├── train/
│   ├── train_readout.py        # CartPole training
│   ├── train_words.py          # NEW – word learning
│   ├── evaluate.py
│   └── evaluate_words.py       # NEW
├── visualize/
│   └── plot_activity.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/Mohamadx2/flybrain-simple-tasks.git
cd flybrain-simple-tasks

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train word association (treat / punishment)

```bash
python -m train.train_words --episodes 2000
```

### Evaluate word learning

```bash
python -m train.evaluate_words
```

### Train CartPole (previous task)

```bash
python -m train.train_readout --episodes 300
python -m train.evaluate --checkpoint checkpoints/best.pt
```

## Design Philosophy

| Part                        | Status                          |
|----------------------------|---------------------------------|
| Circuit structure          | Fixed (inspired by MaleCNS)     |
| Neuron dynamics            | Simple leaky units              |
| Sensory mapping            | Learnable input gains           |
| Decision readout           | **Trained**                     |
| Full synaptic plasticity   | Not included (by design)        |

This matches the honest pattern used in many community fly-connectome projects: keep the circuit structure fixed and only train a lightweight readout.

## Credits & Inspiration

- [awesome-fly](https://github.com/cobanov/awesome-fly) by Mert Cobanov
- [fly-connectome-template](https://github.com/cobanov/fly-connectome-template)
- MaleCNS dataset (Janelia / FlyEM and collaborators)
- Operant conditioning experiments with great apes (for the high-level training idea)

## License

MIT (for the code in this repository).  
Any future use of real MaleCNS data must respect its original CC BY 4.0 license.
