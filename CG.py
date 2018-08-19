__author__ = 'Chih-Hao Fang'
__email__  = 'fang150@purdue.edu'

#################################################
##	This file constains the implementions of   ##
##	Conjugate	Gradient Descent Method        ##
#################################################

import numpy as np
import torch
from torch.autograd import Variable
from softmax import hessian_vector_product


def ConjugateGradient(x,y,W,Lambda,tol,CG_maxit,cols,num_classes,gradient,num_workers,RHO,init_V):
	
	tol2=tol*tol
	g=Variable(torch.DoubleTensor([-1.0]).cuda())*gradient
	V=init_V
	Hv=hessian_vector_product(x,W,V,Lambda,cols,num_classes,RHO)
	r=g-Hv
	h=r
	delta=torch.dot(h.view(num_classes*cols),r.view(num_classes*cols))
	bb=torch.dot(g.view(num_classes*cols),g.view(num_classes*cols))
	p=r
	best_rel_residual=np.array(float("inf"))
	best_V=V
	norm_g=torch.sqrt(torch.dot(g.view(num_classes*cols),g.view(num_classes*cols)))
	norm_r=torch.sqrt(torch.dot(r.view(num_classes*cols),r.view(num_classes*cols)))
	rel_residual=norm_r/norm_g
	iteration=0

	## starts CG iterations 
	for i in range(CG_maxit):

		Hg=hessian_vector_product(x,W,p,Lambda,cols,num_classes,RHO)
		pAp=torch.dot(p.view(num_classes*cols,1),Hg.view(num_classes*cols,1))
		alpha=delta/pAp
		r=r-alpha*Hg
		V=V+alpha*p
		norm_r=torch.sqrt(torch.dot(r.view(num_classes*cols),r.view(num_classes*cols)))
		rel_residual=norm_r/norm_g
		
		## update direction if there's improvement
		if(rel_residual.data.cpu().numpy()<best_rel_residual):
			best_rel_residual=rel_residual.data.cpu().numpy()
			best_V.data=V.data
		h=r
		prev_delta = delta
		delta=torch.dot(h.view(num_classes*cols),r.view(num_classes*cols))
		
		## termination condition
		if(delta.data.cpu().numpy()<tol2*bb.data.cpu().numpy()):
			break
		p=h+(delta/prev_delta)*p
		
		iteration=i
	
	return iteration,best_V,g,norm_g,rel_residual
