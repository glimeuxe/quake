# Quake

by Gregory Lim and Ryan Cheong

[![Dataset | available](https://img.shields.io/badge/GUI-active-red)](https://drive.google.com/file/d/1ln0j21XmYO9onMP6qsE-wtxhP-k7X2w3/view)
|
[![GUI | active](https://img.shields.io/badge/GUI-active-green)](https://huggingface.co/spaces/glimeuxe/quake)

## Setup

1. Clone this repository into a root directory (hereafter, `root`) of your choice.
2. Download the dataset files from [Google Drive](https://drive.google.com/file/d/1ln0j21XmYO9onMP6qsE-wtxhP-k7X2w3/view).
3. Create new directories in `root` as follows.

```
root/
└── dataset/
    └── production/
```

4. Place the downloaded files (`metadata_10000.feather`, `signals_10000.npy`, `spectrogram_images_10000.npy`, `waveform_images_10000.npy`) in `production`:
5. Install the package dependencies. For example, one may run the following command.

```
pip install -r requirements.txt
```

6. Have fun with `main.ipynb` or the the [GUI](https://huggingface.co/spaces/glimeuxe/quake)

## Notes

1. To select dataset processing method `n`, set `PROCESSING_METHOD` to `n`, where `n` is either `1` or `2`.
