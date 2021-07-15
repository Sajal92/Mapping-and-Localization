###############################
#### Loader.py###############
###############################
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread

class couplet(Dataset):

	def __init__(self, path, flag, depth=False, pose=False):

		self.op_path = path+'3Dflow/'
		self.ip_path = path+'rgb/'
		self.depth_path = None
		if depth:
			self.depth_path = path+'depthgt/'

		if flag == 'ev':
			self.img_list = self.scenario_list('0002')
		if flag == 'tr':
			self.img_list = self.scenario_list('0001')
			self.img_list.extend(self.scenario_list('0018'))
			self.img_list.extend(self.scenario_list('0020'))
		if flag == 'te':
			self.img_list = self.scenario_list('0006')
			self.img_list = self.img_list[::2] # halve the data


		def __len__(self):
			return len(self.img_list)

		def __getitem__(self, idx):
			name = self.ip_path+self.img_list[idx][:-3]+'png'
			frame0 = imread(name)
			frame0 = ((frame0/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # ResNet
			frameT = imread(name[0:-9]+((str(int(self.img_list[idx][-9:-4])+100001))[1:]))
			frameT = ((frameT/255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

			disp = np.load(self.op_path+self.img_list[idx])
			disp = disp - self.disp_mean; disp = disp / self.disp_std
			disp = disp/8.5 # this results in prediction and gt to converge to same mean and
			depth0 = None; depthT = None
			if self.depth_path is not None:
				name = self.depth_path+self.img_list[idx][:-3]+'png'
				depth0 = imread(name)
				depth0 = depth0/10000; depth0 = np.clip(depth0,0,2); depth0 -= 1
				depth0 = np.expand_dims(depth0,axis=2)
				depthT = imread(name[0:-9]+((str(int(self.img_list[idx][-9:-4])+100001))[1:]
				depthT = depthT/10000; depthT = np.clip(depthT,0,2); depthT -= 1
				depthT = np.expand_dims(depthT,axis=2)
				depth0 = torch.from_numpy(depth0).type(torch.float).permute((2,0,1))
				depthT = torch.from_numpy(depthT).type(torch.float).permute((2,0,1))

			frame0 = torch.from_numpy(frame0).type(torch.float).permute((2,0,1))
			frameT = torch.from_numpy(frameT).type(torch.float).permute((2,0,1))
			disp = torch.from_numpy(disp).type(torch.float).permute((2,0,1))

			if self.depth_path is not None:
				return {'frame0':frame0, 'frameT':frameT, 'disp_gt':disp, 'depth0':depth0} '
			return {'frame0':frame0, 'frameT':frameT, 'disp_gt':disp}

		def scenario_list(self, scene):

			final = []
			temp = os.listdir(self.op_path+scene+'/clone/'); temp.sort()
			variations = os.listdir(self.op_path+scene+'/'); variations.sort()
			for variation in variations:
				final.extend([scene+'/'+variation+'/'+s for s in temp])
			return final
