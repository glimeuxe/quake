import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torchvision.models import mobilenet_v3_small

class MobileNetV3Small_(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = "MobileNetV3Small"
		self.backbone = mobilenet_v3_small(weights=None)
		self.backbone.classifier = nn.Sequential(
			nn.Linear(self.backbone.classifier[0].in_features, 1),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.backbone(x)

# New units
class SomnialUnitV2(nn.Module):
	def __init__(self, in_channels, k=10, training=True):
		super().__init__()
		self.M = deque(maxlen=k)
		self.delta = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=1),
			nn.Sigmoid()
		)
		self.beta = nn.Sequential(
			nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1), nn.ReLU()
		)

	def forward(self, x_t):
		if self.training:
			self.M.append(x_t.detach())
			x_s = random.choice(self.M) if self.M else x_t
		else:
			x_s = x_t

		if x_s.shape[0] != x_t.shape[0]:
			pad_size = x_t.shape[0] - x_s.shape[0]
			if pad_size > 0:
				x_s = torch.cat([x_s, torch.zeros((pad_size, *x_s.shape[1:]), device=x_t.device)], dim=0)
			else:
				x_s = x_s[:x_t.shape[0]]

		with torch.no_grad():
			diff = (x_t - x_s).abs()
			mask = (diff > diff.mean(dim=(1,2,3), keepdim=True)).float()
		x_s_hat = self.delta(x_s * mask)

		g = self.beta(torch.cat([x_t, x_s_hat], dim=1))
		return g * x_s_hat + (1 - g) * x_t

class SomnialUnit(nn.Module):
	def __init__(self, in_channels, k=10, training=True):
		super().__init__()
		self.M = deque(maxlen=k)
		self.delta = nn.Conv2d(in_channels, in_channels, kernel_size=1)
		self.beta = nn.Sequential(
			nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1), nn.ReLU()
		)

	def forward(self, x_t, training=True):
		if training:
			self.M.append(x_t.detach())
			x_s = random.choice(self.M) if self.M else x_t
		else:
			x_s = x_t

		if x_s.shape[0] != x_t.shape[0]:
			pad_size = x_t.shape[0] - x_s.shape[0]
			if pad_size > 0:
				x_s = torch.cat([x_s, torch.zeros((pad_size, *x_s.shape[1:]), device=x_t.device)], dim=0)
			else:
				x_s = x_s[:x_t.shape[0]]

		x_s_hat = self.delta(x_s)
		g = self.beta(torch.cat([x_t, x_s_hat], dim=1))
		return g * x_s_hat + (1 - g) * x_t

# Convolutional nets
class CNNx(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = "CNNx"
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.su = SomnialUnitV2(64, k=10)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.25)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.su(x)
		x = torch.flatten(x, 1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = self.dropout(F.relu(self.fc3(x)))
		x = torch.sigmoid(self.fc4(x))
		return x

class CNN1(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = "CNN1"
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.25)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = torch.flatten(x, 1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = self.dropout(F.relu(self.fc3(x)))
		x = torch.sigmoid(self.fc4(x))
		return x

class CNN2(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = "CNN2"
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 16)
		self.fc3 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.25)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = torch.flatten(x, 1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = torch.sigmoid(self.fc3(x))
		return x

# Somnial convolutional nets
class SCNN1(nn.Module):
	def __init__(self, training=True):
		super().__init__()
		self.id = "SCNN1"
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.su = SomnialUnit(64, k=10, training=training)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.25)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.su(x)
		x = torch.flatten(x, 1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = self.dropout(F.relu(self.fc3(x)))
		x = torch.sigmoid(self.fc4(x))
		return x

class SCNN2(nn.Module):
	def __init__(self, training=True):
		super().__init__()
		self.id = "SCNN2"
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.su = SomnialUnit(64, k=10, training=training)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 16)
		self.fc3 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.25)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.su(x)
		x = torch.flatten(x, 1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = torch.sigmoid(self.fc3(x))
		return x