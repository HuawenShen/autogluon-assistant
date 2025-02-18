# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial covers the implementation of neural network adapters in SpeechBrain, focusing on memory-efficient fine-tuning techniques like LoRA (Low-Rank Adaptation). It provides code examples for adapter configuration in YAML, manual adapter insertion in training code, and custom adapter implementation (PoolLoRA). The tutorial helps with tasks like configuring adapters for existing models, creating custom adapter architectures, and integrating with the PEFT library. Key features include layer-specific adapter insertion, wildcard-based layer selection, parameter freezing control, and support for both built-in and custom adapter types with configurable parameters like rank and scaling.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Neural Network Adapters Tutorial

## Key Concepts
- Adapters enable memory-efficient fine-tuning by freezing original parameters and adding small trainable parameters
- Low-Rank Adaptation (LoRA) is a popular adapter technique
- SpeechBrain supports both built-in adapters and integration with PEFT library

## Quick Implementation
Basic adapter configuration in YAML:
```yaml
adapted_model: !new:speechbrain.nnet.adapters.AdaptedModel
    model_to_adapt: !ref <model>
    adapter_class: !name:speechbrain.nnet.adapters.LoRA
    all_linear: True
    unfrozen_layers: ["conv_1d_*"]
    adapter_kwargs:
        rank: 8
```

## Implementation Details

### 1. Adding Adapters to Existing Model
1. Create new YAML file with adapter configurations
2. Key modifications needed:
```yaml
new_encoder: !new:speechbrain.nnet.adapters.AdaptedModel
    model_to_adapt: !ref <encoder>
    adapter_class: !name:speechbrain.nnet.adapters.LoRA
    all_linear: True
    manual_adapter_insertion: True
    adapter_kwargs:
        rank: 8
```

### 2. Manual Adapter Insertion
Add to training code:
```python
hparams["pretrainer"].collect_files()
hparams["pretrainer"].load_collected()
hparams["new_encoder"].insert_adapters()
hparams["new_decoder"].insert_adapters()
```

### 3. Custom Adapter Implementation
Example of custom PoolLoRA adapter:
```python
class PoolLoRA(torch.nn.Module):
    def __init__(self, target_module, stride=2, rank=16, alpha=1.0):
        super().__init__()
        # Initialize adapter parameters
        self.adapter_down_scale = torch.nn.AvgPool1d(kernel_size=stride)
        self.adapter_down_proj = torch.nn.Linear(input_size // stride, rank, bias=False)
        self.adapter_up_proj = torch.nn.Linear(rank, output_size, bias=False)
        self.scaling = alpha / rank

    def forward(self, x):
        x_pretrained = self.pretrained_module(x)
        x_downsample = self.adapter_down_proj(self.adapter_down_scale(x))
        x_pool_lora = self.adapter_up_proj(x_downsample)
        return x_pretrained + x_pool_lora * self.scaling
```

## Best Practices & Important Notes
1. Use `manual_adapter_insertion: True` when loading pretrained parameters
2. Adapters can be added to specific layers using `target_layers` argument
3. Use `unfrozen_layers` to specify which layers should remain trainable
4. Supports unix-style wildcards for layer selection through `fnmatch`
5. Can integrate with PEFT library adapters by changing adapter_class

## Configuration Parameters
- `all_linear`: Add adapters to all linear layers
- `all_conv`: Add adapters to all convolution layers
- `rank`: Dimension of low-rank adaptation
- `manual_adapter_insertion`: Control when adapters are inserted
- `adapter_kwargs`: Specific parameters for the chosen adapter type