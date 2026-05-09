# GRPO-Based Continual Learning Benchmark for Agent Tool-Use

A modular benchmark system for evaluating continual learning in tool-using agents, based on GRPO (Group Relative Policy Optimization).

## Overview

This benchmark implements a complete training and evaluation pipeline for studying continual learning in agents that use tools. It supports:

- **Multi-task Sequential Learning**: Train on airline → retail → telecom tasks sequentially
- **GRPO Training**: Generate multiple responses per prompt, compute relative advantages, update policy with true gradients
- **Multi-GPU Data Parallel**: Distribute batches across GPUs using PyTorch DDP
- **Trajectory Recording**: Log all tool calls, responses, and rewards for analysis
- **Oracle Reward**: Use existing tau2-bench evaluators for consistent reward computation
- **Modular CL Algorithms**: Easy to plug in different continual learning algorithms

## Architecture

```
src/tau2/continual_learning/
├── config.py                    # Training configuration (GRPOConfig)
├── data_loader.py               # Task loading from airline/retail/telecom
├── reward_oracle.py             # Reward computation using tau2 evaluators
├── trajectory_buffer.py         # Trajectory storage and replay
├── metrics_tracker.py           # Training metrics and logging
├── policy_model.py              # LLM policy wrapper (TODO)
├── grpo_trainer.py              # Main training loop (TODO)
├── continual_learning/
│   ├── base.py                  # CL algorithm interface
│   └── __init__.py
└── utils/
    ├── distributed.py           # Multi-GPU utilities (TODO)
    └── __init__.py
```

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install accelerate>=0.24.0

# Optional dependencies
pip install flash-attn>=2.3.0  # For flash attention
pip install wandb>=0.15.0      # For experiment tracking
pip install matplotlib>=3.5.0  # For visualization
```

### Install tau2-bench

```bash
cd tau2-bench
pip install -e .
```

## Components Implemented

### ✅ Core Infrastructure

1. **GRPOConfig** (`config.py`)
   - Comprehensive configuration for training
   - Supports open-source models (Qwen, Llama)
   - Multi-GPU settings
   - Validation and defaults

2. **TaskDataLoader** (`data_loader.py`)
   - Loads tasks from airline, retail, telecom domains
   - Train/eval splitting
   - Batch sampling with multiple strategies

3. **RewardOracle** (`reward_oracle.py`)
   - Integrates with tau2-bench evaluators
   - Computes rewards for trajectories
   - Supports multiple evaluation types (ENV, ACTION, ALL)
   - Provides detailed reward breakdown

4. **TrajectoryBuffer** (`trajectory_buffer.py`)
   - Stores trajectories with metadata
   - Multiple sampling strategies (random, high_reward, recent)
   - Persistence (save/load)
   - Statistics and export

5. **MetricsTracker** (`metrics_tracker.py`)
   - Tracks training, evaluation, and transfer metrics
   - Wandb integration
   - Learning curve visualization
   - Summary statistics

6. **CLAlgorithm** (`continual_learning/base.py`)
   - Abstract base class for CL algorithms
   - Three hooks: augment_batch, post_step_hook, post_task_hook
   - SequentialCL baseline implementation

### 🚧 To Be Implemented

The following components need to be implemented to complete the system:

1. **PolicyModel** (`policy_model.py`)
   - Wrapper around open-source LLM (Qwen/Llama)
   - Generate multiple response trajectories
   - Compute log probabilities with gradients
   - GRPO loss computation
   - Policy updates

2. **GRPOTrainer** (`grpo_trainer.py`)
   - Main training loop (task-level, step-level)
   - GRPO algorithm implementation
   - Multi-GPU coordination via DDP
   - Evaluation loop
   - Checkpoint management

3. **Distributed Utilities** (`utils/distributed.py`)
   - Multi-GPU setup helpers
   - Gradient aggregation
   - Batch distribution

4. **Training Script** (`scripts/train_grpo_cl.py`)
   - Entry point for training
   - Argument parsing
   - Trainer initialization

## Usage (Once Complete)

### Basic Training

```python
from tau2.continual_learning import GRPOConfig, TaskDataLoader, GRPOTrainer

# Create configuration
config = GRPOConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    num_samples_per_prompt=4,
    batch_size_per_gpu=4,
    num_steps_per_task=100,
    task_order=["airline", "retail", "telecom"],
    cl_algorithm="sequential"
)

# Create trainer
trainer = GRPOTrainer(config)

# Train
trainer.train()
```

### Multi-GPU Training

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 scripts/train_grpo_cl.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --batch_size_per_gpu 4 \
    --num_steps_per_task 100 \
    --cl_algorithm sequential
```

### Custom CL Algorithm

```python
from tau2.continual_learning.continual_learning import CLAlgorithm

class ReplayCL(CLAlgorithm):
    """Experience replay continual learning."""

    def __init__(self, replay_ratio: float = 0.2):
        self.replay_ratio = replay_ratio

    def augment_batch(self, new_tasks, current_domain):
        # Sample from trajectory buffer
        num_replay = int(len(new_tasks) * self.replay_ratio)

        # Get previous domains
        prev_domains = []
        for domain in self.trainer.config.task_order:
            if domain == current_domain:
                break
            prev_domains.append(domain)

        # Sample replay trajectories
        replay_records = self.trainer.trajectory_buffer.sample_multi_domain(
            domains=prev_domains,
            num_samples_per_domain=num_replay // len(prev_domains) if prev_domains else 0
        )

        # Extract tasks from records
        replay_tasks = [r.task for r in replay_records]

        return new_tasks + replay_tasks

    def post_step_hook(self, trainer, domain):
        pass

    def post_task_hook(self, trainer, domain):
        pass
```

## Configuration Options

### Model Configuration

```python
model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"  # Model path
model_dtype: str = "bfloat16"                          # Precision
temperature: float = 0.7                                # Sampling temperature
max_new_tokens: int = 2048                             # Max generation length
```

### GRPO Hyperparameters

```python
num_samples_per_prompt: int = 4    # Responses per prompt
kl_coef: float = 0.1               # KL divergence penalty
gamma: float = 1.0                 # Discount factor
```

### Training Configuration

```python
batch_size_per_gpu: int = 4                # Batch size per GPU
gradient_accumulation_steps: int = 2       # Gradient accumulation
num_steps_per_task: int = 100              # Steps per task
learning_rate: float = 1e-6                # Learning rate
max_grad_norm: float = 1.0                 # Gradient clipping
```

### Continual Learning

```python
cl_algorithm: str = "sequential"           # CL algorithm
replay_buffer_size: int = 1000             # Buffer capacity
replay_ratio: float = 0.2                  # Replay sample ratio
```

### Task Configuration

```python
task_order: list[str] = ["airline", "retail", "telecom"]  # Task sequence
max_tasks_per_domain: Optional[int] = None                 # Limit tasks
train_split: float = 0.8                                   # Train/eval split
```

## Metrics

The system tracks comprehensive metrics:

### Training Metrics
- Loss per step
- Mean/max reward per step
- Tool selection accuracy

### Evaluation Metrics
- Reward mean/std
- Pass rate (reward > 0.5)
- Tool accuracy

### Transfer Metrics
- **Backward Transfer**: Performance on previous tasks
- **Current Performance**: Performance on current task
- **Average Performance**: Overall performance across all seen tasks

### Buffer Statistics
- Buffer size per domain
- Mean/max/min rewards
- High-reward trajectory count

## Evaluation

The reward oracle uses tau2-bench evaluators to compute rewards based on:

1. **Environment State** (DB): Database state matches expected state
2. **Actions**: Required tool calls were executed correctly
3. **Communication**: Required information was communicated
4. **NL Assertions**: Natural language assertions about agent behavior

Rewards are computed multiplicatively across components specified in `reward_basis`.

## Extending the Benchmark

### Adding New CL Algorithms

1. Create a new class extending `CLAlgorithm`
2. Implement the three hooks:
   - `augment_batch`: Modify training batch
   - `post_step_hook`: Update after each step
   - `post_task_hook`: Consolidate after task completion

3. Register in config:
```python
config = GRPOConfig(cl_algorithm="my_algorithm")
```

### Adding New Domains

1. Add domain to `task_order` in config
2. Ensure domain is registered in tau2-bench registry
3. Add task file path mapping in `TaskDataLoader`

### Custom Metrics

Extend `MetricsTracker` to add custom metrics:

```python
class CustomMetricsTracker(MetricsTracker):
    def log_custom_metric(self, value):
        self.metrics["custom"].append({"value": value})
```

## Implementation Status

### Completed ✅
- Configuration system
- Data loading
- Reward computation
- Trajectory storage
- Metrics tracking
- CL algorithm interface
- Sequential baseline

### In Progress 🚧
- Policy model wrapper
- GRPO trainer
- Distributed utilities
- Training script

### Planned 📋
- ~~Experience replay algorithm~~ ✅ **Implemented!**
- EWC algorithm
- Comprehensive tests
- Example notebooks

## Design Principles

1. **Modularity**: Each component has a clear interface and can be replaced
2. **Extensibility**: Easy to add new CL algorithms, domains, or metrics
3. **Reproducibility**: Full trajectory recording and deterministic evaluation
4. **Scalability**: Multi-GPU support from the start
5. **Compatibility**: Integrates seamlessly with existing tau2-bench infrastructure

## Key Features

### GRPO Algorithm

- Generates multiple responses per prompt
- Computes advantages relative to mean reward within prompt
- Uses KL divergence penalty to prevent policy collapse
- True gradients via backpropagation (not REINFORCE estimators)

### Multi-GPU Training

- PyTorch DDP for data parallelism
- Each GPU processes local batch independently
- Gradients aggregated via AllReduce
- Single parameter update per step

### Trajectory Recording

- Full message history with tool calls
- Reward breakdown by component
- Metadata (domain, task ID, timestamp)
- Export for offline analysis

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{grpo_cl_benchmark,
  title={GRPO-Based Continual Learning Benchmark for Agent Tool-Use},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/tau2-bench}}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
