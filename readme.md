# Quake

[![GUI | inactive](https://img.shields.io/badge/GUI-active-blue)](https://huggingface.co/spaces/glimeuxe/quake)

## Setup

1. Clone this repository into a root directory (hereafter, `root`) of your choice.
2. Download the dataset files from [Hugging Face](https://huggingface.co/datasets/glimeuxe/quake).
3. Download the model directories from [Hugging Face](https://huggingface.co/glimeuxe/quake).
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
7. Optionally, generate and download a raw spectrogram image by running `visualise_raw_data` in `main.ipynb` to upload to the GUI!

## Notes

`main.ipynb` contains the main code. In this file, each function has a simple docstring describing what it does. `architectures.py` contains class definitions for architectures. `utils.py` contains utility functions. To sequentially train and evaluate a model defined by class `M`, do the following.

1. Select a dataset processing method by setting `PROCESSING_METHOD` to either `1` or `2`. Run the first cell.

2. Optionally, run the second cell to visualise annotated waveform and spectrogram images and/or a raw spectrogram image.

3. Replace `model` with an instance of `M` appropriately. Run the third, fourth, fifth, and sixth cells, in order.

4. Optionally, run the seventh cell to generate gradient-weighted class activation mapping (Grad-CAM) overlays for SCNNQ730.
