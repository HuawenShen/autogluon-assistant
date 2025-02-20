# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates implementing hyperparameter optimization in SpeechBrain using the hpopt utility. It covers integrating optimization contexts into existing training code, configuring search spaces through YAML files, and managing optimization trials with Orion. Key implementations include wrapping main training code in hyperparameter optimization contexts, reporting results for optimization, and defining search spaces for parameters like embedding dimensions and TDNN channels. The tutorial helps with tasks like setting up distributed hyperparameter optimization, managing checkpointing during trials, and scaling optimization across multiple GPUs. Essential features include Orion integration, parallel optimization capabilities, and proper handling of hyperparameter configurations in YAML files.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# Hyperparameter Optimization in SpeechBrain

## Key Implementation Steps

1. **Required Imports**
```python
from speechbrain.utils import hpopt as hp
```

2. **Recipe Modifications**

a. Wrap main code in hyperparameter optimization context:
```python
with hp.hyperparameter_optimization(objective_key="error") as hp_ctx:
    hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:])
    # ... existing training code ...
```

b. Add result reporting in `on_stage_end`:
```python
hp.report_result(stage_stats)  # Last reported result used for optimization
```

3. **YAML Configuration**

Add to main `train.yaml`:
```yaml
hpopt_mode: null
hpopt: null
```

Optional `hpopt.yaml` for optimization-specific settings:
```yaml
number_of_epochs: 1
ckpt_enable: false
trial_id: hpopt
output_folder: !ref ./results/speaker_id/<trial_id>
```

## Critical Configurations

1. **Orion Configuration Example**
```yaml
experiment:
    max_trials: 1000
    max_broken: 1000
    algorithms:
        tpe:
            seed: 42
            n_initial_points: 5
```

2. **Search Space Definition**
```bash
orion hunt -n $EXPERIMENT_NAME -c $CONFIG_FILE python train.py train.yaml \
    --hpopt hpopt.yaml \
    --hpopt_mode orion \
    --emb_dim~"choices([128,256,512,768,1024])" \
    --tdnn_channels~"choices([128,256,512,768,1024])"
```

## Important Best Practices

1. **Checkpointer Handling**
- Either disable checkpointing during optimization
- Or make checkpointing conditional:
```python
if self.hparams.ckpt_enable:
    self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])
```

2. **Hyperparameter Accessibility**
- Parameters must be at top level in YAML for optimization
- Use references (!ref) to maintain structure

3. **Scaling Options**
- Supports multi-GPU training (DP/DDP)
- Parallel optimization with `--n-workers` flag
- Distributed optimization across nodes available

## Viewing Results
```bash
orion info --name experiment_name
```

## Warning Notes
- Ensure hyperparameters are at top level in YAML
- Handle checkpointing carefully to avoid architecture conflicts
- Consider using separate directories for different optimization trials