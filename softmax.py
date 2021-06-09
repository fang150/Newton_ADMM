__author__ = 'Chih-Hao Fang'
__email__  = 'fang150@purdue.edu'

#################################################
##	This file constains the implementions of   ##
##  1) softmax evaluation					   ##
##  2) softmax prediction					   ##
##  3) Hessian Vector Product for softmax      ##
################################################# 


import numpy as np
import torch
from torch.autograd import Variable



## softmax fx+regularization term
def Softmax_Fx_Reg(x,y,W,Lambda,cols,num_classes):
	Matrix_Mul=torch.mm(x, W)
	zeros=Variable(torch.DoubleTensor(np.shape(x)[0],1).zero_()).cuda()
	Matrix_concat=torch.cat((Matrix_Mul,zeros), 1)
	Mx=torch.max(Matrix_concat, 1)[0]
	Ax= torch.exp(Variable(torch.DoubleTensor([-1.0]).cuda())*Mx)+torch.sum(torch.exp(Matrix_Mul - Mx.view(np.shape(x)[0],1)),1)
	A1=torch.gather(Matrix_concat,1,y.long()).view(np.shape(x)[0])
	fx=torch.sum(Mx+Ax.log()-A1)
	reg = (Lambda/Variable(torch.DoubleTensor([2.0]).cuda()))*torch.dot(W.view(cols*(num_classes)) ,W.view(cols*(num_classes)))
	return fx+reg


## softmax fx
def Softmax_Fx(x,y,W,Lambda,cols,num_classes):
	Matrix_Mul=torch.mm(x, W)
	zeros=Variable(torch.DoubleTensor(np.shape(x)[0],1).zero_()).cuda()
	Matrix_concat=torch.cat((Matrix_Mul,zeros), 1)
	Mx=torch.max(Matrix_concat, 1)[0]
	Ax= torch.exp(Variable(torch.DoubleTensor([-1.0]).cuda())*Mx)+torch.sum(torch.exp(Matrix_Mul - Mx.view(np.shape(x)[0],1)),1) 
	A1=torch.gather(Matrix_concat,1,y.long()).view(np.shape(x)[0])
	fx=torch.sum(Mx+Ax.log()-A1)
	return fx

## softmax for ADMM
def Softmax_Fx_ADMM(x,y,W,Lambda,cols,num_classes,RHO,Z_term,Y_term):
	Matrix_Mul=torch.mm(x, W)
	zeros=Variable(torch.DoubleTensor(np.shape(x)[0],1).zero_()).cuda()
	Matrix_concat=torch.cat((Matrix_Mul,zeros), 1)
	Mx=torch.max(Matrix_concat, 1)[0]
	Ax= torch.exp(Variable(torch.DoubleTensor([-1.0])).cuda()*Mx)+torch.sum(torch.exp(Matrix_Mul - Mx.view(np.shape(x)[0],1)),1) #torch.exp(-Mx)
	A1=torch.gather(Matrix_concat,1,y.long()).view(np.shape(x)[0])
	fx=torch.sum(Mx+Ax.log()-A1)
	W_shift=Z_term-W+Y_term/RHO
	cost = fx  + (RHO/Variable(torch.DoubleTensor([2.0]).cuda()))*torch.dot(W_shift.view(cols*num_classes) ,W_shift.view(cols*num_classes))
	return cost

## softmax prediction
def predict(x,y,W):
	Matrix_Mul=torch.mm(x, W)
	zeros=Variable(torch.DoubleTensor(np.shape(x)[0],1).zero_()).cuda()
	Matrix_concat=torch.cat((Matrix_Mul,zeros), 1)
	pre_temp1=Variable(torch.DoubleTensor([1.0])).cuda()+torch.sum(torch.exp(Matrix_Mul),1)
	pre_temp2=Variable(torch.DoubleTensor([1.0])).cuda()-torch.sum(torch.exp(Matrix_Mul)/pre_temp1.view((np.shape(x)[0],1)).expand_as(Matrix_Mul),1)
	pre_temp3=torch.exp(Matrix_Mul)/pre_temp1.view((np.shape(x)[0],1)).expand_as(Matrix_Mul)
	pred=torch.cat((pre_temp3,pre_temp2.view((np.shape(x)[0],1)) ),1)
	return pred

## Hessian Vector Product for softmax
def hessian_vector_product(x,W,V,Lambda,cols,num_classes,RHO):
	
	Matrix_Mul=torch.mm(x, W)
	zeros=Variable(torch.DoubleTensor(np.shape(x)[0],1).zero_(),volatile=True).cuda()
	Matrix_concat=torch.cat((Matrix_Mul,zeros), 1)
	Mx=torch.max(Matrix_concat, 1)[0]
	Ax= torch.exp(Variable(torch.DoubleTensor([-1.0]).cuda())*Mx)+torch.sum(torch.exp(Matrix_Mul - Mx.view(np.shape(x)[0],1).expand_as(Matrix_Mul)),1) 
	B=torch.exp(Matrix_Mul - Mx.view(np.shape(x)[0],1).expand_as(Matrix_Mul))/Ax.view(np.shape(x)[0],1).expand_as(Matrix_Mul)
	A=torch.mm(x, V)
	C=A*B-B*torch.mm(torch.mm(A*B, Variable(torch.ones(num_classes,1).double().cuda()) ), Variable(torch.ones(1,num_classes).double().cuda()) ) 
	Hv= torch.mm(torch.transpose(x, 0, 1),C)+RHO*V
	return Hv
