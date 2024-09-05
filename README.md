# Differentiable Channel Selection in Self-Attention For Person Re-Identification

## Search and re-train DCS model

#### Person Re-ID

Search

```python
python tools/search.py --config_file='configs/reid_DAS_config.yml'MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('your path to save checkpoints and logs')"	DATASETS.ROOT_DIR "('Root directory where datasets should be used')"
```

Train

```python
python tools/train.py --config_file='configs/reid_DAS_config.yml'MODEL.PRETRAIN_CHOICE "('self')" MODEL.PRETRAIN_PATH "('your path to pretrained weights')" MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('your path to save checkpoints and logs')"	DATASETS.ROOT_DIR "('Root directory where datasets should be used')"
```

