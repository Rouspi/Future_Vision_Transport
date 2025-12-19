# Module `fvt.utils`

Rôle : utilitaires (métriques, pertes, helpers MLflow, etc.).

Fichiers clés :
- `losses.py` : fonctions de pertes (`ce_weighted`, `dice`, `ce_dice`), métriques (DiceMetric, mIoU).
- `mlflow_utils.py` : fonctions d’aide pour MLflow (tracking URI, récupération de run ID, etc.).
- `models.py` : helpers divers liés aux modèles.

Notes :
- Les custom_objects pour charger un modèle Keras incluent souvent `DiceMetric` et la loss correspondante (via `build_loss`).
- Assurez-vous d’avoir une version TF/Keras compatible avec le format du modèle (Keras 3 si sauvegardé avec TF 2.17+).
