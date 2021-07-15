###############################
#### Loss.py###############
###############################

import numpy as np
import torch
import torch.nn as nn

class LossWeighted(nn.Module):

	def __init__(self):
		super(LossWeighted, self).__init__()
		self.original = nn.MSELoss(reduction='none')

	def forward(self, gt, pred):
		mask = gt.abs().sign()
		c = (mask.sum())/(mask.numel())
		mask = (1-2*c)*mask + c
		loss = torch.mul(self.original(gt,pred),mask)
		loss = (loss.sum())/(mask.sum())
		return loss
