# FlyBrain Simple Tasks

Train simple readouts on a **fruit-fly (MaleCNS-inspired) mapped neural circuit** for basic tasks.

This is an educational starter project inspired by the [awesome-fly](https://github.com/cobanov/awesome-fly) collection and the [fly-connectome-template](https://github.com/cobanov/fly-connectome-template).

> **Important**: This is **not** a full biological brain simulation.  
> We use a small, fixed circuit structure inspired by real MaleCNS connectivity patterns, then train only a lightweight readout (or a few gains) for simple tasks.

## What you can do

- Train a small neural readout on top of a fixed fly-inspired circuit
- Run simple tasks (CartPole-style balancing, binary classification, simple control)
- Visualize activity of the mapped neurons
- Keep the connectome structure frozen and only train the readout (common pattern in the community)

## Project Structure

```
flybrain-simple-tasks/
├── circuit/                 # Fixed fly-inspired circuit definition
│   └── male_cns_subset.py
├── tasks/                   # Simple environments / tasks
│   ├── cartpole_lite.py
│   └── classification.py
├── train/                   # Training scripts
│   ├── train_readout.py
│   └── evaluate.py
├── visualize/               # Simple activity plots
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

# Train a simple readout on the CartPole-lite task
python -m train.train_readout --task cartpole --episodes 300

# Evaluate
python -m train.evaluate --task cartpole --checkpoint checkpoints/best.pt
```

## Design Philosophy

Following patterns used in many awesome-fly projects:

| Part                        | Status                          |
|----------------------------|---------------------------------|
| Circuit structure          | Fixed (inspired by MaleCNS)     |
| Neuron dynamics            | Simple LIF / leaky units        |
| Sensory mapping            | Hand-designed / learnable gains |
| Motor / decision readout   | **Trained**                     |
| Full synaptic plasticity   | Not included (by design)        |

This keeps experiments honest and computationally light while still being grounded in real connectome ideas.

## Credits & Inspiration

- [awesome-fly](https://github.com/cobanov/awesome-fly) by Mert Cobanov
- [fly-connectome-template](https://github.com/cobanov/fly-connectome-template)
- MaleCNS dataset (Janelia / FlyEM and collaborators)
- Community projects such as Fly Dino, fly-cartpole, Flyhard, etc.

## License

MIT (for the code in this repository).  
Any future use of real MaleCNS data must respect its original CC BY 4.0 license.
