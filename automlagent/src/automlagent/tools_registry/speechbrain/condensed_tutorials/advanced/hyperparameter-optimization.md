# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates implementing hyperparameter optimization in SpeechBrain using Orion, covering essential techniques for automated parameter tuning in speech processing models. It helps with tasks like configuring and executing hyperparameter searches, integrating optimization contexts into existing training code, and managing experiment results. Key features include YAML-based configuration setup, optimization context implementation, result reporting mechanisms, checkpointing strategies, and distributed training support. The tutorial provides practical code snippets for dependency installation, recipe modifications, configuration file setup, and experiment monitoring, making it valuable for implementing systematic hyperparameter optimization in speech processing applications.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Hyperparameter Optimization in SpeechBrain

## Key Implementation Steps

1. **Install Dependencies**
```bash
pip install orion[profet]
pip install pyyaml==5.4.1
```

2. **Recipe Modifications**

Import required module:
```python
from speechbrain.utils import hpopt as hp
```

Wrap main code with optimization context:
```python
with hp.hyperparameter_optimization(objective_key="error") as hp_ctx:
    hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:])
    # ... existing training code ...
```

3. **Report Results**
```python
# Add in on_stage_end when stage == sb.Stage.VALID
hp.report_result(stage_stats)
```

4. **YAML Configuration**

Add to train.yaml:
```yaml
hpopt_mode: null
hpopt: null
```

Optional hpopt.yaml for optimization-specific settings:
```yaml
number_of_epochs: 1
ckpt_enable: false
trial_id: hpopt
output_folder: !ref ./results/speaker_id/<trial_id>
```

## Critical Configurations

1. **Orion Configuration File**
```yaml
experiment:
    max_trials: 1000
    max_broken: 1000
    algorithms:
        tpe:
            seed: 42
            n_initial_points: 5
```

2. **Launch Command**
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
- Supports multi-GPU training via DP/DDP
- Parallel optimization with `--n-workers` flag
- Distributed optimization across nodes available

## Monitoring Results
```bash
orion info --name experiment_name
```

This shows:
- Completed trials
- Best objective value
- Optimal hyperparameter values
- Experiment duration