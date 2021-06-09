__author__ = 'Chih-Hao Fang'
__email__  = 'fang150@purdue.edu'

#################################################
##	This file constains the implementions of   ##
##	Backtracking-Armijo LineSearch             ##
#################################################


import numpy as np
import torch
from torch.autograd import Variable
from softmax import Softmax_Fx_ADMM


def LineSearch(d,X,Y,W,Lambda,LS_maxit,gradient,cols,num_classes,RHO,Z_term,Y_term):
	rho=Variable(torch.DoubleTensor([0.5])).cuda()
	c=Variable(torch.DoubleTensor([0.0001])).cuda()
	alphak=Variable(torch.DoubleTensor([1.0])).cuda()
	s=W
	fk=Softmax_Fx_ADMM(X,Y,s,Lambda,cols,num_classes,RHO,Z_term,Y_term)
	gk=gradient
	ss = s
	s = s + alphak*d
	fk1=Softmax_Fx_ADMM(X,Y,s,Lambda,cols,num_classes,RHO,Z_term,Y_term)
	temp=torch.dot(gk.view(num_classes*cols),d.view(num_classes*cols))

	iteration=0

	while((fk+c*alphak*temp).data.cpu().numpy()<fk1.data.cpu().numpy()):

		if(iteration>LS_maxit):
			break
		alphak=alphak*rho
		s=ss
		s = s + alphak*d
		fk1=Softmax_Fx_ADMM(X,Y,s,Lambda,cols,num_classes,RHO,Z_term,Y_term)
		iteration+=1
	return alphak,iteration
