# Module `fvt.models`

Rôle : définitions des architectures de segmentation.

Architectures supportées (via `TrainingConfig.model_type`) :
- `mobilenetv2_deeplab_lite`
- `unet_small`
- `vgg16_unet`

Usage (via pipeline) :
```python
from fvt.training.pipeline import train_segmentation_model
from fvt.training.config import TrainingConfig
from fvt.config import load_settings

cfg = TrainingConfig(run_name="demo_run", model_type="unet_small")
settings = load_settings()
train_segmentation_model(cfg, settings)
```

Notes :
- Les dimensions d’entrée doivent correspondre à `input_height`/`input_width`.
- Les poids de classes et la loss se règlent via `TrainingConfig` (`loss_type`, `class_weights`).
