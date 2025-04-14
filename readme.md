# Quake

## Setup

1. Create a root directory (e.g. `root`) with this structure:

```
root/
└── dataset/
    └── production/
```

2. Place these files in `dataset/production/`:
   - `metadata_10000.feather`
   - `signals_10000.npy`
   - `spectrogram_images_10000.npy`
   - `waveform_images_10000.npy`

3. Place these files in `root/`:
   - `main.ipynb`
   - `utils.py`
   - `architectures.py`

4. Install dependencies:

```
pip install -r requirements.txt
```

## Notes

- Python ≥ 3.8 is recommended.
- A download link for the data will be provided later.
