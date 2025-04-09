import gc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from pathlib import Path
from torch.utils.data import Dataset

def seed_functions(seed):
	"""Seeds functions from numpy and torch."""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def plot_spectrogram(trace, sampling_rate=100, img_width=3, img_height=2, dpi=100):
	"""Generates a spectrogram image from a given trace signal with a fixed output size."""
	fig, ax = plt.subplots(figsize=(img_width, img_height), dpi=dpi)

	# Plot the spectrogram
	ax.specgram(trace, Fs=sampling_rate, NFFT=256, cmap="gray", vmin=-10, vmax=25)
	ax.set_xlim([0, 60])
	ax.axis("off")  # Hide axis labels

	# Save to buffer
	buf = BytesIO()
	fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
	buf.seek(0)

	# Convert buffer to image and force fixed size
	img = Image.open(buf).convert("RGB")  # Ensure 3-channel format
	img = img.resize((300, 200))  # Resize to (width=300, height=200)
	img_arr = np.array(img)

	plt.close(fig)  # Free memory
	return img_arr

def plot_waveform(trace, img_width=3, img_height=1, dpi=100):
	"""Generates a waveform image from a given trace signal with a fixed output size."""
	fig, ax = plt.subplots(figsize=(img_width, img_height), dpi=dpi)

	# Generate x-axis values
	x = np.linspace(0, 60, len(trace))

	# Plot waveform
	ax.plot(x, trace, color="k", linewidth=1)
	ax.set_xlim([0, 60])
	ax.axis("off")  # Hide axis labels

	# Save to buffer
	buf = BytesIO()
	fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
	buf.seek(0)

	# Convert buffer to image and force fixed size
	img = Image.open(buf).convert("RGB")  # Ensure 3-channel format
	img = img.resize((300, 100))  # Resize to (width=300, height=100)
	img_arr = np.array(img)

	plt.close(fig)  # Free memory
	return img_arr


def add_gaussian_noise(trace, noise_level_range=(0.001, 0.01)):  ## Adds Gaussian noise to a trace
	"""Adds random Gaussian noise to a trace. Noise level is relative to trace std dev."""
	trace_std = np.std(trace, axis=0)  ## Standard deviation per channel
	noise_factor = np.random.uniform(*noise_level_range, size=trace.shape[1])  ## Random scale per channel
	noise = np.random.randn(*trace.shape) * trace_std * noise_factor  ## Generate noise
	noisy_trace = trace + noise  ## Add noise
	return noisy_trace

class DataPreprocessing:
	"""Handles data preprocessing, including metadata reading, subsampling, and waveform loading."""
	def __init__(self, subsample_size, raw_dataset_path, logs_path, noise_ratio, earthquake_ratio):
		# Store initialization parameters
		self.subsample_size = subsample_size
		self.raw_dataset_path = raw_dataset_path
		self.logs_path = logs_path
		self.noise_ratio = noise_ratio
		self.earthquake_ratio = earthquake_ratio

		# Fetch data file paths
		self._fetch_datapaths_from_dir()

		# Read metadata from CSV files
		self._parse_metadata_csvs()

		# Create a balanced subsample of the dataset
		self.subsample_metadata = self._get_balanced_subsample()

		# Read waveform data from HDF5 files
		subsample_trace_names = list(self.subsample_metadata.index)
		self.subsample_traces = self._read_h5py_files(subsample_trace_names)

		# Keep only metadata entries for successfully loaded traces
		self.subsample_metadata = self.subsample_metadata.loc[self.subsample_traces.keys()]

	def _fetch_datapaths_from_dir(self):
		"""Finds all metadata and data files in the dataset directory."""
		self.metadata_paths = []
		self.data_paths = []
		for category in ["noise", "earthquake"]:
			# Construct category path
			category_path = os.path.join(self.raw_dataset_path, category)

			# Collect all CSV and HDF5 file paths
			self.metadata_paths.extend(Path(category_path).rglob("*.csv"))
			self.data_paths.extend(Path(category_path).rglob("*.hdf5"))

	def _parse_metadata_csvs(self):
		"""Reads and merges metadata from multiple CSV files."""
		metadata_dfs = []
		for path in self.metadata_paths:
			# Read CSV file
			df = pd.read_csv(path, low_memory=False)

			# Assign category label based on file path
			df["category"] = "earthquake" if "earthquake" in str(path) else "noise"

			metadata_dfs.append(df)

		# Combine all metadata into a single DataFrame
		self.full_metadata = pd.concat(metadata_dfs).reset_index(drop=True)

	def _get_balanced_subsample(self):
		"""Creates a balanced subsample of noise and earthquake signals."""
		# Sample earthquake traces
		earthquake_samples = self.full_metadata[self.full_metadata["category"] == "earthquake"].sample(
			int(self.subsample_size * self.earthquake_ratio), random_state=0
		)

		# Sample noise traces
		noise_samples = self.full_metadata[self.full_metadata["category"] == "noise"].sample(
			int(self.subsample_size * self.noise_ratio), random_state=0
		)

		# Merge and shuffle the subsample
		subsample_metadata = pd.concat([earthquake_samples, noise_samples]).sample(frac=1, random_state=0)

		# Set index to trace names
		subsample_metadata.set_index("trace_name", drop=True, inplace=True)
		return subsample_metadata

	def _read_h5py_files(self, trace_names):
		"""Reads waveform traces from HDF5 files."""
		traces = {}
		for path in self.data_paths:
			with h5py.File(path, "r") as h5f:
				for trace_name in trace_names:
					# Retrieve trace data
					trace = np.array(h5f.get(f"data/{trace_name}"))

					# Print shape for debugging
					print(f"Trace '{trace_name}' shape: {trace.shape}")

					# Skip invalid or empty traces
					if trace is None or trace.shape == () or trace.size == 0:
						print(f"Skipping invalid trace '{trace_name}'")
						with open(os.path.join(self.logs_path, "skipped_traces.log"), "a") as log_file:
							log_file.write(f"{trace_name}\n")
						continue

					traces[trace_name] = trace
		return traces

	def create_waveform_images(self):
		"""Generates waveform images for all subsampled traces."""
		waveform_imgs = np.zeros((len(self.subsample_traces), 100, 300, 3), dtype=np.uint8)
		for i, (trace_name, trace) in enumerate(self.subsample_traces.items()):
			try:
				# Add Gaussian noise only to earthquake traces  ##
				if self.subsample_metadata.loc[trace_name, "category"] == "earthquake":  ##
					trace = add_gaussian_noise(trace)  ## Apply Gaussian noise

				# Print trace statistics before plotting
				print(f"Waveform Trace '{trace_name}': min={np.min(trace)}, max={np.max(trace)}, mean={np.mean(trace)}")

				# Generate waveform image
				waveform_imgs[i] = plot_waveform(trace[:, 2])  # Use the third channel
			except Exception as e:
				print(f"Failed to process waveform '{trace_name}': {e}")  # Debugging output

			# Free memory periodically
			if i % 100 == 0:
				gc.collect()

		return waveform_imgs

	def create_spectrogram_images(self):
		"""Generates spectrogram images for all subsampled traces."""
		spectrogram_imgs = np.zeros((len(self.subsample_traces), 200, 300, 3), dtype=np.uint8)
		for i, (trace_name, trace) in enumerate(self.subsample_traces.items()):
			try:
				# Add Gaussian noise only to earthquake traces  ##
				if self.subsample_metadata.loc[trace_name, "category"] == "earthquake":  ##
					trace = add_gaussian_noise(trace)  ## Apply Gaussian noise

				# Print trace statistics before plotting
				print(f"Spectrogram Trace '{trace_name}': min={np.min(trace)}, max={np.max(trace)}, mean={np.mean(trace)}")

				# Generate spectrogram image
				spectrogram_imgs[i] = plot_spectrogram(trace[:, 2])  # Use the third channel
			except Exception as e:
				print(f"Failed to process spectrogram '{trace_name}': {e}")  # Debugging output

			# Free memory periodically
			if i % 100 == 0:
				gc.collect()

		return spectrogram_imgs

class SpectrogramDataset(Dataset):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		return self.images[i], self.labels[i]