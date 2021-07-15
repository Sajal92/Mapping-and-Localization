###############################
#### Network0.py###############
###############################
import torch.nn as nn
import torch
from torchvision.models import resnet18

class Net(nn.Module):

	def __init__(self):

		super(Net, self).__init__()
		resnet = resnet18(pretrained=True)
		for child in list(resnet.children())[:-2]:
			for param in child.parameters():
				param.requires_grad = False

		self.down = nn.Sequential(*list(resnet.children())[:-2]) # 12, 39

		self.up = nn.Sequential(
		nn.Conv2d(1024, 256, 3, padding=1, bias=True),
		nn.BatchNorm2d(256),
		nn.ReLU(inplace=True),
		# nn.Upsample(scale_factor=4,mode='bicubic'),
		nn.Conv2d(256, 64, 3, padding=1, bias=True),
		nn.BatchNorm2d(64),
		nn.ReLU(inplace=True),
		# nn.Upsample(scale_factor=2,mode='bicubic'),
		nn.Conv2d(64, 16, 3, padding=1, bias=True),
		nn.BatchNorm2d(16),
		nn.ReLU(inplace=True),
		nn.Upsample(scale_factor=3,mode='bicubic', align_corners=False), # 36, 1
		nn.Conv2d(16, 8, 3, padding=1, bias=True),
		nn.BatchNorm2d(8),
		nn.ReLU(inplace=True),
		nn.Upsample(size=(125,414),mode='bicubic', align_corners=False), # 125,
		nn.Conv2d(8, 3, 3, padding=1, bias=True),
		nn.BatchNorm2d(3),nn.Upsample(scale_factor=3,mode='bicubic', align_corners=False) # 375, 1
		)

	def forward(self, frame0, frameT):

		frame0 = self.down(frame0)
		frameT = self.down(frameT)

		frame0 = torch.cat((frame0, frameT), 1)
		frame0 = self.up(frame0)

		return frame0;
