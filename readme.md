Let `root` be a root directory of your choice.

1. Download and place `main.ipynb` and `utils.py` in `root`.
2. Ensure you have the following directories prepared in `root`:

`.` (`root`)

-> `dataset`

--> `production`

--> `raw`

---> `earthquake`

----> `chunk6.csv`

----> `chunk6.hdf5`

---> `noise`

----> `chunk1.csv`

----> `chunk1.hdf5`

-> `logs`

-> `models`

where `chunk6.csv`, `chunk6.hdf5`, `chunk1.csv`, and `chunk1.hdf5` are downloaded from https://github.com/smousavi05/STEAD, and the directories not containing those files are empty. (Alternatively, to reproduce our results, download and place our dataset files `metadata_10000.feather`, `signals_10000.npy`, `spectrogram_images_10000.npy`, and `waveform_images_10000.npy` in `production`.)
