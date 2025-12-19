# Module `fvt.inference`

Rôle : prédiction/inférence à partir d’un modèle entraîné.

Fichiers clés :
- `predict.py` : fonctions de prédiction sur des images (chemins locaux).

Usage typique :
```python
import tensorflow as tf
from fvt.utils.losses import DiceMetric, build_loss

model = tf.keras.models.load_model(
    "artifacts/models/<run_name>/model.keras",
    custom_objects={
        "DiceMetric": DiceMetric,
        build_loss("ce_dice").__name__: build_loss("ce_dice")
    },
)
# Prétraiter vos images au format attendu (H, W)
# preds = model.predict(batch)
```

Notes :
- Respecter la taille d’entrée du modèle (cf. TrainingConfig).
- Ajouter les `custom_objects` correspondant à la loss/métrique utilisée à l’entraînement.
