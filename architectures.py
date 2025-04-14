import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torchvision.models import mobilenet_v3_small
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from neuralop.models import FNO

# New units
class SomnialUnit(nn.Module):
	def __init__(self, in_channels, k=10):
		super().__init__()
		self.M = deque(maxlen=k)
		self.generator = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=1),
			nn.Identity()
		)

	def reshape_memory_sample(self, x_s, x_t):
		if x_s.shape[0] != x_t.shape[0]:
			pad_size = x_t.shape[0] - x_s.shape[0]
			if pad_size > 0:
				x_s = torch.cat([x_s, torch.zeros((pad_size, *x_s.shape[1:]), device=x_t.device)], dim=0)
			else:
				x_s = x_s[:x_t.shape[0]]
		return x_s

	def modulator(self, x_s_hat, x_t):
		q = F.normalize(x_t, dim=1)
		k = F.normalize(x_s_hat, dim=1)
		return torch.sigmoid((q * k).sum(dim=1, keepdim=True))

	def forward(self, x_t):
		if self.training:
			self.M.append(x_t.detach())
			x_s = random.choice(self.M)
		else:
			x_s = x_t
		x_s = self.reshape_memory_sample(x_s, x_t)
		x_s_hat = self.generator(x_s)
		m = self.modulator(x_s_hat, x_t)
		return m * x_s_hat + (1 - m) * x_t

# Imported nets
class SR50ViTB16(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.expects224 = True
		self.backbone = timm.create_model("vit_base_r50_s16_224", pretrained=False, num_classes=0)
		self.backbone_stem = self.backbone.patch_embed.backbone
		in_channels = self.backbone_stem.feature_info[-1]['num_chs']
		self.somnial = SomnialUnit(in_channels)
		self.pool = nn.Identity()
		self.classifier = nn.Sequential(
			nn.Linear(self.backbone.num_features, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.backbone_stem(x)
		x = self.somnial(x)
		x = self.backbone.patch_embed.proj(x)
		x = x.flatten(2).transpose(1, 2)
		if hasattr(self.backbone, 'pos_drop'):
			x = self.backbone.pos_drop(x + self.backbone.pos_embed[:, :x.size(1)])
		for blk in self.backbone.blocks:
			x = blk(x)
		if hasattr(self.backbone, 'norm'):
			x = self.backbone.norm(x)
		x = x[:, 0]
		x = self.pool(x)
		x = self.classifier(x)
		return x

class R50ViTB16(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.expects224 = True
		self.backbone = timm.create_model("vit_base_r50_s16_224", pretrained=False, num_classes=0)
		self.pool = nn.Identity()
		self.classifier = nn.Sequential(
			nn.Linear(self.backbone.num_features, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.backbone(x)

		x = self.pool(x)
		x = self.classifier(x)
		return x

class R26ViTS32(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.expects224 = True
		self.backbone = timm.create_model("vit_small_r26_s32_224", pretrained=False, num_classes=0)
		self.pool = nn.Identity()
		self.classifier = nn.Sequential(
			nn.Linear(self.backbone.num_features, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.backbone(x)
		x = self.pool(x)
		x = self.classifier(x)
		return x

class ViTB16(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.expects224 = True
		self.backbone = vit_b_16(weights=None)
		self.backbone.heads = nn.Sequential(
			nn.Linear(self.backbone.heads.head.in_features, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.backbone(x)

class MNV3S(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.backbone = mobilenet_v3_small(weights=None)
		self.backbone.classifier = nn.Sequential(
			nn.Linear(self.backbone.classifier[0].in_features, 1),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.backbone(x)

# Convolutional nets
class CNNV1Q(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
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

class CNNV1H(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 16)
		self.fc3 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.5)

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
class SCNNV1Q(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.su = SomnialUnit(64, k=10)
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

class SCNNV1H(nn.Module):
	def __init__(self):
		super().__init__()
		self.id = self.__class__.__name__
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.su = SomnialUnit(64, k=10)
		self.fc1 = nn.Linear(64 * 25 * 37, 128)  # 59200
		self.fc2 = nn.Linear(128, 16)
		self.fc3 = nn.Linear(16, 1)
		self.dropout = nn.Dropout(0.5)

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