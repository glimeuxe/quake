# Quake

[![Dataset | available](https://img.shields.io/badge/Dataset-available-red)](https://drive.google.com/file/d/1ln0j21XmYO9onMP6qsE-wtxhP-k7X2w3/view)
[![Models | available](https://img.shields.io/badge/Models-available-red)](https://huggingface.co/spaces/glimeuxe/quake/tree/main/models)
[![GUI | active](https://img.shields.io/badge/GUI-active-blue)](https://huggingface.co/spaces/glimeuxe/quake)

## Setup

1. Clone this repository into a root directory (hereafter, `root`) of your choice.
2. Download the dataset files from [Google Drive](https://drive.google.com/file/d/1ln0j21XmYO9onMP6qsE-wtxhP-k7X2w3/view).
3. Download the model directories from [Hugging Face](https://huggingface.co/spaces/glimeuxe/quake/tree/main/models).
4. Create (empty) directories in `root` as follows.

```
root/
└── dataset/
    └── production/
└── models/
```

4. Place the downloaded dataset files (`metadata_10000.feather`, `signals_10000.npy`, `spectrogram_images_10000.npy`, and `waveform_images_10000.npy`) in `production`. Place the downloaded model directories (`SR50ViTB16` and `SCNNQ730`) in `models`. The directories in `root` should resemble the following.

```
root/
└── dataset/
    └── production/
        └── metadata_10000.feather
        └── signals_10000.npy
        └── spectrogram_images_10000.npy
        └── waveform_images_10000.npy
└── models/
    └── SCNNQ730
        └── best.pth
        └── losses.json
        └── output.png
    └── SR50ViTB16
        └── best.pth
        └── losses.json
        └── output.png
```

6. Install the package dependencies. For example, one may run the following command in the terminal.

```
pip install -r requirements.txt
```

6. Play with `main.ipynb`!
7. Optionally, generate and download a raw spectrogram image by running `visualise_raw_data` in `main.ipynb` or from [Google Drive](https://drive.google.com/file/d/1cPLsMM9ucBGQMaqjy_Mw1ubo9vSX94RO/view) to upload to the [GUI](https://huggingface.co/spaces/glimeuxe/quake)!

## Notes
1. To select dataset processing method `n`, set `PROCESSING_METHOD` to `n`, where `n` is either `1` or `2`.
2. `preprocess_data` is what we used to generate the dataset files from the original Stanford EArthquake Dataset (STEAD). (This function need not run to reproduce our results.)
3. `visualise_data` is what we use to plot annotated waveform and spectrogram images from the dataset files. (This function need not run to reproduce our results.)
4. `visualise_raw_data` is what we use to plot a raw spectrogram image from the dataset files. (This function need not run to reproduce our results.)
5. `split_data` is what we use to obtain index-label pairs needed to split the processed dataset.
6. `load_data` is what we use to split the processed dataset, and load the splits into dataloaders for training a model.
