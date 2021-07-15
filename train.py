import torch, tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from loss import LossWeighted
from loader import couplet
import networks.network1 as network

criterion = nn.MSELoss(reduction='mean') # criterion = LossWeighted()
datapath = '../maploc/vkitti_1.3.1_'
batch_size = 8; log_nth = 10; epochs = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
name = network.__name__.split('.')[1]
writer = SummaryWriter(comment=name)

depth = False; pose = False
if int(name[-1]) == 2: depth = True
if int(name[-1]) == 4: pose = True
tr_set = couplet(datapath,'tr',depth=depth,pose=pose)
ev_set = couplet(datapath,'ev',depth=depth,pose=pose)
train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=5)
eval_loader = DataLoader(ev_set, batch_size=batch_size, shuffle=True, num_workers=5)

def train(model, epochNum, depth, pose):

	tr_loss = []
	train_batching = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
	for batch_i, data in train_batching:
		optimizer.zero_grad()
		frame0 = data['frame0'].to(device); frameT = data['frameT'].to(device)
		disp_gt = data['disp_gt'].to(device)
		if depth:
			depth0 = data['depth0'].to(device); depthT = data['depthT'].to(device)
			disp_pred = model(frame0, frameT, depth0, depthT)
		else: disp_pred = model(frame0, frameT)

		loss = criterion(disp_gt, disp_pred)
		loss.backward()

		optimizer.step()
		tr_loss.append(loss.detach().item())

		if (batch_i+1) % log_nth == 0:
			train_batching.set_description(f'Train E: {epoch+1}, B: {batch_i+1}')
		writer.add_scalar('Loss/train', np.mean(tr_loss), epoch)

		torch.save(model, f"{name}.model")

def evaluate(model, epoch, best_eval, scheduler, depth, pose):

	model.eval()
	with torch.no_grad():
		ev_loss = []
		eval_batching = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
		for batch_i, data in eval_batching:
			frame0 = data['frame0'].to(device); frameT = data['frameT'].to(device)
			disp_gt = data['disp_gt'].to(device)
			if depth:
				depth0 = data['depth0'].to(device); depthT = data['depthT'].to(device)
				disp_pred = model(frame0, frameT, depth0, depthT)
			else: disp_pred = model(frame0, frameT)
			loss = criterion(disp_gt, disp_pred)
			ev_loss.append(loss.detach().item())
		loss = np.mean(ev_loss)
		print(f'\nEval E: {epoch+1}, L: {loss:.2E}\n')
		if best_eval is None or loss < best_eval:
			best_eval = loss
			torch.save(model, f"best_{name}.model")
		writer.add_scalar('Loss/eval', np.mean(ev_loss), epoch)

	scheduler.step()
	return best_eval

model = network.Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
best_eval = None

for epoch in range(epochs):
	train(model, epoch, depth, pose)
	best_eval = evaluate(model, epoch, best_eval, scheduler, depth, pose)
