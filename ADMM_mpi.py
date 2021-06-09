__author__ = 'Chih-Hao Fang'
__email__  = 'fang150@purdue.edu'


#################################################
##	This file has the implementation of        ## 
##  Newton-ADMM 				               ##
##  with Spectral Penalty Parameter Selection  ##        
#################################################

import o
import sys
import time
import numpy as np

import torch
import torch.distributed as dist
from torch.autograd import Variable

from softmax import *
from CG import *
from LS import *



Lambda=Variable(torch.DoubleTensor([np.float64(sys.argv[1])])).cuda()
CG_tol=np.float64(sys.argv[2])
CG_maxit=np.int(sys.argv[3])

Newton_maxit=np.int(sys.argv[4])
LS_maxit=np.int(sys.argv[5])
data_path= sys.argv[6]
num_nodes = int(sys.argv[7])


def accuracy(y,y_pred):
	equal_count=0
	for i in range(y.shape[0]):
		if(y[i]==y_pred[i]):
			equal_count+=1
	
	return (equal_count/float(y.shape[0]))*100

def get_test_stats(X_test,Y_test,cols,num_classes,Z_term,Lambda):
	x2=Variable(torch.from_numpy(X_test).double(),requires_grad=False).cuda()#torch.randn(N, cols)
	y2=Variable(torch.unsqueeze(torch.from_numpy(Y_test).double(),1),requires_grad=False).cuda()			
	total_cost=0
	total_accu=0
	test_values, test_indices=torch.max(predict(x2,y2,Z_term), 1)
	test_accu=accuracy(y2.view(y2.shape[0]).data.cpu().numpy(),test_indices.data.cpu().numpy())
	test_cost=Softmax_Fx_Reg(x2,y2,Z_term,Lambda,cols,num_classes)
	del x2,y2
	return test_accu,test_cost/X_test.shape[0]

def get_train_stats_master(Lambda,Z_term,cols,num_classes,size,group):
	total_cost_tensor=torch.DoubleTensor([0.0])
	total_accu_tensor=torch.DoubleTensor([0.0])
	total_len_tensor=torch.DoubleTensor([0.0])
	dist.reduce(total_cost_tensor,0,dist.reduce_op.SUM,group)
	dist.reduce(total_accu_tensor,0,dist.reduce_op.SUM,group)
	dist.reduce(total_len_tensor,0,dist.reduce_op.SUM,group)
	total_cost=total_cost_tensor
	train_cost=total_cost+((Lambda/Variable(torch.DoubleTensor([2.0]).cuda()))*torch.dot(Z_term.view(cols*(num_classes)) ,Z_term.view(cols*(num_classes)))).data.cpu()
	train_accu=total_accu_tensor/(size-1)
	return train_accu,train_cost/total_len_tensor.numpy()[0]

def compute_train_stats_slave(x,y,Z_new,Lambda,cols,num_classes,group):
	train_values, train_indices=torch.max(predict(x,y,Z_new), 1)
	total_accu=accuracy(y.view(y.shape[0]).data.cpu().numpy(),train_indices.data.cpu().numpy())
	total_cost=Softmax_Fx(x,y,Z_new,Lambda,cols,num_classes)
	total_cost_tensor=total_cost.data.cpu()
	total_accu_tensor=torch.DoubleTensor([total_accu])
	total_len_tensor=torch.DoubleTensor([x.size()[0]])
	dist.reduce(total_cost_tensor,0,dist.reduce_op.SUM,group)
	dist.reduce(total_accu_tensor,0,dist.reduce_op.SUM,group)
	dist.reduce(total_len_tensor,0,dist.reduce_op.SUM,group)


def init_lists(cols,num_classes,size):
	Z_Y_update_list=[]
	

	for index in range(size):
		Z_Y_update_list.append(torch.DoubleTensor(2*cols*num_classes).zero_())		

	All_list=[]
	for index in range(size):
		All_list.append(torch.DoubleTensor(2*cols*num_classes+1).zero_())

	return Z_Y_update_list,All_list

def init_All_master(cols,num_classes,size):
	W=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda()  , requires_grad=True)
	Z_term=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())
	Y_term=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())

	Z_Y_update_list,All_list=init_lists(cols,num_classes,size)
	return W,Z_term,Y_term,Z_Y_update_list,All_list

def init_All_slave(cols,num_classes,size):
	W_old=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda()  , requires_grad=True)
	W_cur=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda()  , requires_grad=True)
	W_new=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda()  , requires_grad=True)
	W_k0=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())
	Z_old=Variable( torch.DoubleTensor(cols, num_classes).zero_()).cuda()
	Z_cur=Variable( torch.DoubleTensor(cols, num_classes).zero_()).cuda()
	Z_new=Variable( torch.DoubleTensor(cols, num_classes).zero_()).cuda()
	Z_k0=Variable( torch.DoubleTensor(cols, num_classes).zero_()).cuda()
	Y_cur=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda(), requires_grad=True)
	Y_new=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda(), requires_grad=True)
	Y_old=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda(), requires_grad=True)
	Y_t_cur=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())
	Y_t_new=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())
	Y_t_k0=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())
	Y_k0=Variable( torch.DoubleTensor(cols, num_classes).zero_().cuda())

	rho_old=Variable(torch.DoubleTensor([0.001]).cuda())
	rho_cur=Variable(torch.DoubleTensor([0.001]).cuda())
	rho_new=Variable(torch.DoubleTensor([0.001]).cuda())
	update_T=1

	W_Y_rho=torch.DoubleTensor(2*cols*num_classes+1).zero_()
	Z_Y_Term_Temp=torch.DoubleTensor(2*cols*num_classes).zero_()

	return W_old,W_cur,W_new,W_k0,\
		   Z_old,Z_cur,Z_new,Z_k0,\
		   Y_cur,Y_new,Y_old,Y_k0,\
		   Y_t_cur,Y_t_new,Y_t_k0,\
		   rho_old,rho_cur,rho_new,update_T,\
		   W_Y_rho,Z_Y_Term_Temp


def update_rho(update_T,epoch,cols,num_classes,W_cur,W_new,W_k0,Y_cur,Y_new,Y_old,Y_t_new,Y_k0,Y_t_k0,Z_cur,Z_new,Z_old,Z_k0,rho_cur,rho_new,rho_old):

	if(epoch%update_T==0):
		Y_t_new.data=Y_cur.data+rho_cur.data*(Z_cur.data-W_new.data)
		delta_u=(W_new.data-W_k0.data).view(cols*num_classes)
		delta_y_t=(Y_t_new.data-Y_t_k0.data).view(cols*num_classes)
		alpha_sd=torch.dot(delta_y_t,delta_y_t)/torch.dot(delta_u,delta_y_t)
		alpha_mg=torch.dot(delta_u,delta_y_t)/torch.dot(delta_u,delta_u)

		delta_z=  (Z_k0.data-Z_new.data).view(cols*num_classes)
		delta_y= (Y_new.data-Y_k0.data).view(cols*num_classes)

		if(torch.dot(delta_z,delta_y)==0):
			beta_sd=torch.dot(delta_y,delta_y)/1e-26
		else:
			beta_sd=torch.dot(delta_y,delta_y)/torch.dot(delta_z,delta_y)
		beta_mg=torch.dot(delta_z,delta_y)/torch.dot(delta_z,delta_z)

		if(2*alpha_mg>alpha_sd):
			a_k=alpha_mg
		else:
			a_k=alpha_sd-alpha_mg/2
		if(2*beta_mg>beta_sd):
			b_k=beta_mg
		else:
			b_k=beta_sd-beta_mg/2

		alpha_cor=torch.dot(delta_u,delta_y_t)/(torch.norm(delta_u)*torch.norm(delta_y_t))

		if(torch.dot(delta_z,delta_y)==0):
			beta_cor=0
		else:
			beta_cor=torch.dot(delta_z,delta_y)/(torch.norm(delta_z)*torch.norm(delta_y))
		
		rho_temp=Variable(torch.DoubleTensor([0.0]).cuda())
		rho_temp1=Variable(torch.DoubleTensor([0.0]).cuda())
		
		if(alpha_cor>0.2 and beta_cor >0.2 ):
			rho_temp.data=torch.sqrt(torch.DoubleTensor([a_k*b_k]).cuda())
		elif(alpha_cor>0.2 and beta_cor <=0.2):
			rho_temp.data=torch.sqrt(torch.DoubleTensor([a_k]).cuda())
		elif(alpha_cor<=0.2 and beta_cor >0.2):
			rho_temp.data=torch.sqrt(torch.DoubleTensor([b_k]).cuda())
		else:
			rho_temp.data=rho_cur.data

		if(rho_temp.data.cpu().numpy()>(1.0+1e10/((epoch+1)*(epoch+1)))*rho_cur.data.cpu().numpy()):
			rho_temp1.data=(1.0+1e10/((epoch+1)*(epoch+1)))*rho_cur.data
		else:
			rho_temp1.data=rho_temp.data

		if(rho_temp1.data.cpu().numpy()>rho_cur.data.cpu().numpy()/(1.0+1e10/((epoch+1)*(epoch+1))) ):
			rho_new.data=rho_temp1.data
		else:
			rho_new.data=rho_cur.data/(1.0+1e10/((epoch+1)*(epoch+1)))

		W_k0.data=W_cur.data
		Z_k0.data=Z_cur.data
		Y_t_k0.data=Y_old.data+rho_old.data*(Z_old.data-W_cur.data)
		Y_k0.data=Y_cur.data
		
	else:
		rho_new.data=rho_cur.data


def update_Z(All_list,cols,num_classes):
	temp=torch.DoubleTensor(cols*num_classes).zero_()
	temp_1=torch.DoubleTensor([0.0]).zero_()

	for index in range(size):	
		if(index==0):
			continue
		else:
			temp+=All_list[index][2*cols*num_classes:2*cols*num_classes+1]*All_list[index][0:cols*num_classes]-All_list[index][cols*num_classes:2*cols*num_classes]
			temp_1+=All_list[index][2*cols*num_classes:2*cols*num_classes+1]

	return ( (1/(Lambda.data.cpu()+ temp_1)) * (temp).view(cols,num_classes)  ).cuda()
	
def update_Z_Y_list(All_list,Z_Y_update_list,Z_term,cols,num_classes):
	for index in range(size):
		if(index==0):
			continue
		else:
			Z_Y_update_list[index][0:cols*num_classes]=Z_term.view(cols*num_classes).data.cpu()
			Z_Y_update_list[index][cols*num_classes:2*cols*num_classes]=All_list[index][cols*num_classes:2*cols*num_classes]+ All_list[index][2*cols*num_classes:2*cols*num_classes+1]*(Z_term.data.view(cols*num_classes).cpu()-All_list[index][0:cols*num_classes])
	

def run_Newton_ADMM(rank, size,Lambda,CG_tol,CG_maxit,data_path,Newton_maxit,LS_maxit):

	## Master load Testing set
	if(dist.get_rank()==0):

		X_test = np.load(data_path+"test_mat.npy" )
		Y_test = np.load(data_path+"test_vec.npy"  )
		Y_test=Y_test.astype(np.float64)
		X_test=X_test.astype(np.float64)

		if(np.min(Y_test)==1):
			Y_test=Y_test-1.0

		cols, num_classes =  X_test.shape[1], int(np.max(Y_test))

	## Slave load Training set
	else:
		X_train = np.load(data_path+"train_mat_split_"+str(int(dist.get_rank()-1))+".npy" )
		Y_train = np.load(data_path+"train_vec_split_"+str(int(dist.get_rank()-1))+".npy"  )
		Y_train= Y_train.astype(np.float64)
		X_train= X_train.astype(np.float64)
		if(np.min(Y_train)==1):
			Y_train=Y_train-1.0
		cols, num_classes = X_train.shape[1], int(np.max(Y_train))

		x=Variable(torch.from_numpy(X_train).double(),requires_grad=False).cuda()
		y=Variable( torch.unsqueeze(torch.from_numpy(Y_train).double(),1),requires_grad=False).cuda()

	## Master: Starts Newton-ADMM routine
	if(dist.get_rank()==0):

		group = dist.new_group(range(size))

		W,Z_term,Y_term,Z_Y_update_list,All_list=init_All_master(cols,num_classes,size)		
		
		cumul_t=0
		total_optimal_cost=0

		## Starts Newton-ADMM iterations
		for epoch in range(Newton_maxit):

			start_time = time.time()
			## receive x_k+1 yk rho_k from slaves
			torch.distributed.gather(torch.DoubleTensor(2*cols*num_classes+1).zero_(),gather_list=All_list,group=group)
			
			## update z 
			Z_term.data=update_Z(All_list,cols,num_classes) 
			update_Z_Y_list(All_list,Z_Y_update_list,Z_term,cols,num_classes)


			## beging statistics
			start_time_fx =time.time()
			test_accu,test_cost=get_test_stats(X_test,Y_test,cols,num_classes,Z_term,Lambda)
			## broadcast updated z and y
			dist.scatter(torch.DoubleTensor(2*cols*num_classes).zero_(),scatter_list=Z_Y_update_list,group=group)
			train_accu,train_cost=get_train_stats_master(Lambda,Z_term,cols,num_classes,size,group)

			end_time_fx = time.time()
			time_fx=(end_time_fx-start_time_fx)
			end_time = time.time()
			cumul_t+=(end_time-start_time-time_fx)*1000
			print("%9d \t %3.2f \t %e \t %3.2f \t %e \t %4.2f \n "% \
				( (epoch+1),train_accu,train_cost,test_accu,test_cost,cumul_t ))

			## end of statistics

	## Slave: Starts Newton-ADMM routine
	else:

		group = dist.new_group(range(size))

		W_old,W_cur,W_new,W_k0,\
		Z_old,Z_cur,Z_new,Z_k0,\
		Y_cur,Y_new,Y_old,Y_k0,\
		Y_t_cur,Y_t_new,Y_t_k0,\
		rho_old,rho_cur,rho_new,update_T,\
		W_Y_rho,Z_Y_Term_Temp = init_All_slave(cols,num_classes,size)
		


		## Starts Newton-ADMM iterations
		for epoch in range(Newton_maxit):

			## compute local gradients
			cost=Softmax_Fx_ADMM(x,y,W_cur,Lambda,cols,num_classes,np.asscalar(rho_cur.data.cpu().numpy()),Z_cur,Y_cur)
			cost.backward()
			
			## obtain local direction using CG
			CG_iter,local_dir,g,norm_g,rel_residual=ConjugateGradient(x,y,W_cur,Lambda,CG_tol,CG_maxit,cols,num_classes,W_cur.grad,num_nodes,np.asscalar(rho_cur.data.cpu().numpy()),Variable(  torch.DoubleTensor(cols, num_classes).zero_().cuda(),volatile=True ))
			

			## obtain step size using backtracking Line Search 
			step_size,ls_iter=LineSearch(local_dir,x,y,W_cur,Lambda,LS_maxit,g,cols,num_classes,np.asscalar(rho_cur.data.cpu().numpy()),Z_cur,Y_cur)

			## updates x_k+1
			W_new.data= W_cur.data + step_size.data*local_dir.data
			
			
			W_Y_rho[0:cols*num_classes]=W_new.data.view(cols*num_classes).cpu()
			W_Y_rho[cols*num_classes:2*cols*num_classes]=Y_cur.data.view(cols*num_classes).cpu()
			W_Y_rho[2*cols*num_classes:2*cols*num_classes+1]=rho_cur.data.cpu()
			
			## send x_k+1, y_k, rho_k to master
			torch.distributed.gather(W_Y_rho,dst=0,group=group)

			## get updated y , z from master 
			dist.scatter(Z_Y_Term_Temp,src=0,group=group)

			Z_new.data=Z_Y_Term_Temp[0:cols*num_classes].view(cols,num_classes).cuda()
			Y_new.data=Z_Y_Term_Temp[cols*num_classes:2*cols*num_classes].view(cols,num_classes).cuda()

			## compute training stats using updated parameters locally
			## and sent them to master 
			compute_train_stats_slave(x,y,Z_new,Lambda,cols,num_classes,group)


			## update rho using Spectral Penalty Parameter Selection Strategy
			update_rho(update_T,epoch,cols,num_classes,\
						W_cur,W_new,W_k0,\
						Y_cur,Y_new,Y_old,Y_t_new,Y_k0,Y_t_k0,\
						Z_cur,Z_new,Z_old,Z_k0,\
						rho_cur,rho_new,rho_old)

			W_cur.grad.data.zero_()
			W_old.data=W_cur.data
			W_cur.data=W_new.data
			Y_old.data=Y_cur.data
			Y_cur.data=Y_new.data
			Z_old.data=Z_cur.data
			Z_cur.data=Z_new.data
			rho_old.data=rho_cur.data
			rho_cur.data=rho_new.data

def init_processes(rank, size, fn, Lambda,CG_tol,CG_maxit,data_path,Newton_maxit,LS_maxit,backend):
	""" Initialize the distributed environment. """
	dist.init_process_group(backend, rank=rank, world_size=size)
	fn(rank, size,Lambda,CG_tol,CG_maxit,data_path,Newton_maxit,LS_maxit)


if __name__ == "__main__":
	size = num_nodes
	init_processes(0, size, run_Newton_ADMM, Lambda,CG_tol,CG_maxit,data_path,Newton_maxit,LS_maxit,'mpi')


