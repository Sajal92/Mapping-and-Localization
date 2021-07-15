###############################
#### Network1.py###############
###############################
import torch.nn as nn
import torch
from torchvision.models import resnet18

class Net(nn.Module):

	def __init__(self):

		super(Net, self).__init__()
		resnet = resnet18(pretrained=True)
		for child in list(resnet.children())[0:8]:
			for param in child.parameters():
				param.requires_grad = False

		self.down0 = nn.Sequential(*list(resnet.children())[0:3])
		self.down1 = nn.Sequential(*list(resnet.children())[3:5])
		self.down2 = list(resnet.children())[5]
		self.down3 = list(resnet.children())[6]
		self.down4 = list(resnet.children())[7] # 12*39
		self.up4 = nn.Sequential(
		nn.Conv2d(1024, 256, 3, padding=1, bias=True),
		nn.BatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False) # 24*78
		)
		self.up3 = nn.Sequential(
		nn.Conv2d(256*3, 128, 3, padding=1, bias=True),
		nn.BatchNorm2d(128),
		nn.ReLU(inplace=True),
		nn.Upsample(size=(47,156),mode='bicubic',align_corners=False) # 47*156
		)
		self.up2 = nn.Sequential(
		nn.Conv2d(128*3, 64, 3, padding=1, bias=True),
		nn.BatchNorm2d(64),
		nn.ReLU(inplace=True),
		nn.Upsample(size=(94,311),mode='bicubic',align_corners=False) # 94*311
		)
		self.up1 = nn.Sequential(
		nn.Conv2d(64*3, 64, 3, padding=1, bias=True),
		nn.BatchNorm2d(64),
		nn.ReLU(inplace=True),
		nn.Upsample(size=(188,621),mode='bicubic',align_corners=False) # 188*621
		)
		self.up0 = nn.Sequential(
		nn.Conv2d(64*3, 32, 3, padding=1, bias=True),
		nn.BatchNorm2d(32),
		nn.ReLU(inplace=True),
		nn.Upsample(size=(375,1242),mode='bicubic',align_corners=False), # 375*1
		nn.Conv2d(32, 8, 3, padding=1, bias=True),
		nn.BatchNorm2d(8),
		nn.ReLU(inplace=True),
		nn.Conv2d(8, 3, 3, padding=1, bias=False),
		nn.BatchNorm2d(3)
		)

	def forward(self, frame0, frameT):

		frame0 = self.down0(frame0)
		frame0_1 = self.down1(frame0)
		frame0_2 = self.down2(frame0_1)
		frame0_3 = self.down3(frame0_2)
		frame0_4 = self.down4(frame0_3)

		frameT = self.down0(frameT)
		frameT_1 = self.down1(frameT)
		frameT_2 = self.down2(frameT_1)
		frameT_3 = self.down3(frameT_2)
		frameT_4 = self.down4(frameT_3)




		frame0_4 = torch.cat((frame0_4, frameT_4), 1)
		frame0_4 = self.up4(frame0_4)
		frame0_3 = torch.cat((frame0_4, frame0_3, frameT_3), 1)
		frame0_3 = self.up3(frame0_3)
		frame0_2 = torch.cat((frame0_3, frame0_2, frameT_2), 1)
		frame0_2 = self.up2(frame0_2)
		frame0_1 = torch.cat((frame0_2, frame0_1, frameT_1), 1)
		frame0_1 = self.up1(frame0_1)
		frame0 = torch.cat((frame0_1, frame0, frameT), 1)
		frame0 = self.up0(frame0)

		return frame0;
