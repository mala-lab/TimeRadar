# TimeRadar for anomaly detection

This directory contains the pretrained TimeRadar checkpoint in Hugging Face
custom-model format. The model accepts tensors shaped as
`[batch, sequence_length, channels]`; this checkpoint requires a sequence
length of 100, while the channel count may vary.

## Dependencies

```bash
pip install transformers==4.40.1 torch-frft==0.8.1 safetensors
```

## Load and run

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./TimeRadar",
    trust_remote_code=True,
)
model.eval()

input_values = torch.randn(8, 100, 25)
with torch.no_grad():
    outputs = model(input_values=input_values)

print(outputs.anomaly_scores.shape)   # (8, 100)
print(outputs.reconstructions.shape)  # (10, 8, 100, 25)
```

Inference uses random symmetric masks. Reset `torch.manual_seed(...)` before a
call when reproducible scores are required. `copies` must be even when using
the default `mask_mode="c"`.

