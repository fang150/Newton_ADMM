# Newton ADMM

This project has the implementation of Newton ADMM for our paper 

[Newton-ADMM: A Distributed GPU-Accelerated Optimizer for Multiclass Classification Problems](https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap476.html)

## Environment

We implemented Newton-ADMM on top of Pytorch with mpi support.

Currently, if user wants Pytorch with mpi support, one needs to install Pytorch from source.

For the instructions of installing it from source, please see [here](https://github.com/pytorch/pytorch#from-source)


## Running the Script

The main python script that runs Newton-ADMM is ADMM_mpi.py.

It takes the following 7 arguments:
1) Lambda         :  The regularization parameter.
2) CG_tol         :  The tolerance of Congugate Gradient(CG) Descent Method
3) CG_maxit       :  The maximun number of CG iterations
4) Newton_maxit   :  The maximun number of Newton iterations, that is, epoch.
5) LS_maxit       :  The maximun number of Line Search Method iterations
6) data_path      :  The path of BOTH your training set and testing set.
7) num_nodes      :  The number of nodes(including Master and Workers) to run Newton-ADMM.

For example

```
mpirun -np num_nodes -hostfile yourhostfile  python ADMM_mpi.py Lambda CG_tol CG_maxit Newton_maxit LS_maxit data_path num_nodes
```

Specifically, if a user wants 1 Master and 2 workers to run Newton-ADMM, the above command would be

```
mpirun -np 3 -hostfile yourhostfile  python ADMM_mpi.py Lambda CG_tol CG_maxit Newton_maxit LS_maxit data_path 3
```

And the format/content of "yourhostfile" in this case is:

```
master_host_name
worker1_host_name
worker2_host_name
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


