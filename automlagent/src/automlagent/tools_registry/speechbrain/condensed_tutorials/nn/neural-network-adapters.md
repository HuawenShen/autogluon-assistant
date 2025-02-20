# Condensed: <!-- This cell is automatically updated by tools/tutorial-cell-updater.py -->

Summary: This tutorial demonstrates implementing neural network adapters for efficient model fine-tuning, focusing on LoRA and custom adapter architectures in SpeechBrain. It covers configuration techniques for selective layer adaptation using YAML configs, manual adapter insertion procedures, and implementation of custom adapters like PoolLoRA. The tutorial helps with tasks involving memory-efficient model fine-tuning, reducing trainable parameters to ~1% while maintaining performance. Key features include configurable adapter ranks, layer freezing controls, integration with PEFT library, and custom adapter development with detailed parameter management and scaling mechanisms.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Neural Network Adapters for Faster Low-Memory Fine-tuning

## Key Implementation

### Basic Configuration
```yaml
adapted_model: !new:speechbrain.nnet.adapters.AdaptedModel
    model_to_adapt: !ref <model>
    adapter_class: !name:speechbrain.nnet.adapters.LoRA
    all_linear: True
    unfrozen_layers: ["conv_1d_*"]
    adapter_kwargs:
        rank: 8
```

### Important Parameters
- `all_linear`: Adds adapters to all linear layers
- `all_conv`: Adds adapters to all convolution layers
- `unfrozen_layers`: Specifies which layers should remain trainable
- `target_layers`: Specifies specific layers for adaptation
- Supports unix-style wildcards through `fnmatch`

## Implementation Details

### Manual Adapter Insertion
When loading pretrained parameters:
1. Set `manual_adapter_insertion: True` in YAML
2. Add insertion code after loading pretrained parameters:
```python
hparams["pretrainer"].collect_files()
hparams["pretrainer"].load_collected()
hparams["new_encoder"].insert_adapters()
hparams["new_decoder"].insert_adapters()
```

### Custom Adapter Example
```python
class PoolLoRA(torch.nn.Module):
    def __init__(self, target_module, stride=2, rank=16, alpha=1.0):
        super().__init__()
        input_size = target_module.weight.data.shape[1]
        output_size = target_module.weight.data.shape[0]
        
        # Freeze pretrained module
        self.pretrained_module = target_module
        for param in self.pretrained_module.parameters():
            param.requires_grad = False
            
        # Adapter components
        self.adapter_down_scale = torch.nn.AvgPool1d(kernel_size=stride)
        self.adapter_down_proj = torch.nn.Linear(input_size // stride, rank, bias=False)
        self.adapter_up_proj = torch.nn.Linear(rank, output_size, bias=False)
        self.adapter_up_proj.weight.data.fill_(0.0)
        self.scaling = alpha / rank

    def forward(self, x):
        x_pretrained = self.pretrained_module(x)
        x_downsample = self.adapter_down_proj(self.adapter_down_scale(x))
        x_pool_lora = self.adapter_up_proj(x_downsample)
        return x_pretrained + x_pool_lora * self.scaling
```

## Best Practices
1. Use adapters for memory-efficient fine-tuning of large models
2. Configure adapter rank based on memory/performance tradeoff needs
3. Carefully select which layers to freeze/unfreeze
4. Consider custom adapters for specific architectural needs

## Integration Options
- Use SpeechBrain's built-in adapters
- Integrate PEFT library adapters
- Implement custom adapters as needed

This implementation typically reduces trainable parameters to ~1% of original model size while maintaining performance.