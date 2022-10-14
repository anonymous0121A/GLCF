from copyreg import pickle
from statistics import mean
from sys import stdout
from time import time
from tkinter import mainloop
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import infoNCE, KLDiverge, pairPredict, calcRegLoss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, teacher):
		super(Model, self).__init__()

		# self.teacher = LightGCN()
		self.teacher = teacher
		self.student = MLPNet()
	
	def forward(self):
		pass

	def adaptivePck(self, ancs, uEmbeds, iEmbeds, top):
		# pckIdims = t.randint(args.item, [args.topRange])
		# iEmbeds = iEmbeds[pckIdims]
		allPreds = self.teacher.predAll(uEmbeds[ancs], iEmbeds)
		_, topLocs = t.topk(allPreds, args.topRange, largest=top)
		pckLocs = t.randint(args.topRange, [topLocs.shape[0], 1]).cuda()
		return t.gather(topLocs, 1, pckLocs)

	def calcLoss(self, adj, ancs, poss, negs, opt):
		uniqAncs = t.unique(ancs)
		uniqPoss = t.unique(poss)
		suEmbeds, siEmbeds = self.student()
		tuEmbeds, tiEmbeds = self.teacher(adj)
		tuEmbeds = tuEmbeds.detach()
		tiEmbeds = tiEmbeds.detach()

		rdmUsrs = t.randint(args.user, [args.topRange])#ancs
		rdmItms1 = t.randint_like(rdmUsrs, args.item)#.view([-1, 1])
		rdmItms2 = t.randint_like(rdmUsrs, args.item)#.view([-1, 1])

		# contrastive regularization for node embeds
		tEmbedsLst = self.teacher(adj, getMultOrder=True)
		highEmbeds = sum([tEmbedsLst[3]])
		highuEmbeds = highEmbeds[:args.user].detach()
		highiEmbeds = highEmbeds[args.user:].detach()
		contrastDistill = (infoNCE(highuEmbeds, suEmbeds, uniqAncs, args.tempcd) + infoNCE(highiEmbeds, siEmbeds, uniqPoss, args.tempcd)) * args.cdreg


		# soft-target-based distillation
		tpairPreds = self.teacher.pairPredictwEmbeds(tuEmbeds, tiEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		# tpairPreds = self.teacher.pairPredictwEmbeds(highEmbeds[:args.user], highEmbeds[args.user:], rdmUsrs, rdmItms1, rdmItms2)
		spairPreds = self.student.pairPredictwEmbeds(suEmbeds, siEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		softTargetDistill = KLDiverge(tpairPreds, spairPreds, args.tempsoft) * args.softreg

		# structural contrastive


		# main task
		# preds = self.student.pairPredictwEmbeds(suEmbeds, siEmbeds, ancs, poss, negs)
		# mainLoss = -(preds).sigmoid().log().mean()

		preds = self.student.pointPosPredictwEmbeds(suEmbeds, siEmbeds, ancs, poss)
		mainLoss = -t.log(t.sigmoid(preds)).mean()

		opt.zero_grad()
		(contrastDistill).backward(retain_graph=True)
		pckUGrad1 = suEmbeds.grad[ancs].detach().clone()
		pckIGrad1 = siEmbeds.grad[poss].detach().clone()
		opt.zero_grad()
		(softTargetDistill).backward(retain_graph=True)
		pckUGrad2 = suEmbeds.grad[ancs].detach().clone()
		pckIGrad2 = siEmbeds.grad[poss].detach().clone()
		opt.zero_grad()
		mainLoss.backward(retain_graph=True)
		pckUGrad4 = suEmbeds.grad[ancs].detach().clone()
		pckIGrad4 = siEmbeds.grad[poss].detach().clone()
		def calcGradSim(grad1, grad2, grad4):
			grad3 = grad1 + grad2
			sim12 = (grad1 * grad2).sum(-1)
			sim34 = (grad3 * grad4).sum(-1)
			return (sim34 < sim12) * 0.0 + (sim34 > sim12) * 1.0
		uTaskGradSim = calcGradSim(pckUGrad1, pckUGrad2, pckUGrad4)
		iTaskGradSim = 0#calcGradSim(pckIGrad1, pckIGrad2, pckIGrad4)

		# selfContrast = (t.log(self.student.pointNegPredictwEmbeds(suEmbeds, siEmbeds, ancs, args.tempsc, uTaskGradDissim) + 1e-5).mean() * 1e-3 + t.log(self.student.pointNegPredictwEmbeds(suEmbeds, suEmbeds, ancs, args.tempsc, uTaskGradDissim) + 1e-5).mean() * 1e-3 + t.log(self.student.pointNegPredictwEmbeds(siEmbeds, siEmbeds, poss, args.tempsc, iTaskGradDissim) + 1e-5).mean() * 3e-3)
		selfContrast = (t.log(self.student.pointNegPredictwEmbeds(siEmbeds, siEmbeds, poss, args.tempsc) + 1e-5) * (1 - iTaskGradSim)).mean() * args.screg


		# weight-decay reg
		regParams = [self.student.uEmbeds, self.student.iEmbeds]
		regLoss = calcRegLoss(params=regParams) * args.sreg
		# regLoss = calcRegLoss(model=self) * args.sreg

		loss = mainLoss + contrastDistill + softTargetDistill + regLoss + selfContrast
		losses = {'mainLoss': mainLoss, 'contrastDistill': contrastDistill, 'softTargetDistill': softTargetDistill, 'regLoss': regLoss}
		return loss, losses
	
	def testPred(self, usr, trnMask, adj=None):
		if adj is None:
			uEmbeds, iEmbeds = self.student()
		else:
			uEmbeds, iEmbeds = self.teacher(adj)
		allPreds = t.mm(uEmbeds[usr], t.transpose(iEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
		return allPreds

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.linear1 = nn.Linear(args.latdim, args.latdim, bias=False)
		self.act = nn.LeakyReLU(negative_slope=0.5)
		# self.linear2 = nn.Linear(args.latdim, 1, bias=False)

	def forward(self, inp):
		tem = self.act(self.linear1(inp)) + inp
		# ret = self.linear2(tem).view(-1)
		ret = tem.sum(dim=-1)
		return ret

class MLPNet(nn.Module):
	def __init__(self):
		super(MLPNet, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.MLP = MLP()
	
	def forward(self):
		return self.uEmbeds, self.iEmbeds
	
	def pointPosPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		# return t.exp((ancEmbeds * posEmbeds).sum(-1))
		nume = self.MLP(ancEmbeds * posEmbeds)
		return nume
	
	def pointNegPredictwEmbeds(self, embeds1, embeds2, nodes1, temp=1.0):
		pckEmbeds1 = embeds1[nodes1]

		return t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)


		num = embeds2.shape[0]
		minibat = num // 100
		predsSum = 0
		for i in range(num // minibat + (num % minibat!=0)):
			curEmbeds2 = embeds2[i * minibat: (i + 1) * minibat, :]
			mlpinp = t.einsum('bd, nd -> bnd', pckEmbeds1, curEmbeds2).view([-1, args.latdim])
			curPreds = self.MLP(mlpinp).view([nodes1.shape[0], curEmbeds2.shape[0]])
			predsSum += t.exp(curPreds).sum(-1)
		return predsSum
		mlpinp = t.einsum('bd, nd -> bnd', pckEmbeds1, embeds2).view([-1, args.latdim])
		preds = self.MLP(mlpinp).view([nodes1.shape[0], embeds2.shape[0]])
		return t.exp(preds).sum(-1)
	
	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		posPreds = self.MLP(ancEmbeds * posEmbeds)
		negPreds = self.MLP(ancEmbeds * negEmbeds)
		return posPreds - negPreds
	
	def predAll(self, pckUEmbeds, iEmbeds):
		mlpinp = t.einsum('bd, nd -> bnd', pckUEmbeds, iEmbeds).view([-1, args.latdim])
		return self.MLP(mlpinp).view([pckUEmbeds.shape[0], iEmbeds.shape[0]])
	
	def testPred(self, usr, trnMask):
		uEmbeds, iEmbeds = self.forward()
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds

class LightGCN(nn.Module):
	def __init__(self):
		super(LightGCN, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward(self, adj, getMultOrder=False):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)
		if not getMultOrder:
			return embeds[:args.user], embeds[args.user:]
		else:
			return embedsLst
	
	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		return pairPredict(ancEmbeds, posEmbeds, negEmbeds)
	
	def predAll(self, pckUEmbeds, iEmbeds):
		return pckUEmbeds @ iEmbeds.T
	
	def testPred(self, usr, trnMask, adj):
		uEmbeds, iEmbeds = self.forward(adj)
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)


class MF(nn.Module):
	def __init__(self):
		super(MF, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
	
	def forward(self):
		return self.uEmbeds, self.iEmbeds

	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		# print(ancEmbeds.shape, posEmbeds.shape)
		return pairPredict(ancEmbeds, posEmbeds, negEmbeds)
	
	def predAll(self, pckUEmbeds, iEmbeds):
		return pckUEmbeds @ iEmbeds.T
	
	def testPred(self, usr, trnMask):
		uEmbeds, iEmbeds = self.forward()
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds
