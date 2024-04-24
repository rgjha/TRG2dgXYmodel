#####################################
###### 2d Generalized XY Model ######
#####################################

######### modules #########
import sys
from time import time, ctime
from torch.cuda import is_available as check_cuda
from scipy.special import iv
from math import sqrt, log
from numpy import prod

## check if CUDA is available on the device ##
use_cuda = check_cuda()

## load architecture specific python modules ##
if not use_cuda:
    print("ARCHITECTURE : Central Processing Unit (CPU)")
    from opt_einsum import contract
    from numpy import max, reshape, transpose
    from scipy.linalg import svd
else:
    print("ARCHITECTURE : Graphical Processing Unit (GPU)")
    from torch import reshape as Treshape
    from torch import tensor, permute
    from torch import max as Tmax
    from torch.linalg import svd as Tsvd
    from opt_einsum_torch import einsum

## model and algorithm parameter values ##
Temp =  float(sys.argv[1])
h =  float(sys.argv[2])             
h1 = float(sys.argv[3])
D = int(sys.argv[4])
Niters = int(sys.argv[5])
delta = float(sys.argv[6])

if len(sys.argv) != 6:
    ## exit running the code if the number of parameters ## 
    ## in the input are not specified properly ##
    print("ERROR : invalid number of parameters assigned!\nplease specify the following:")
    print("Temperature\nh-field value\nh1-field value\nBond Dimension\nNumber of Iterations\nDeformation Parameter")
    sys.exit(1)

## subsequent derived quantities for the simulation ##
beta = float(1.0/Temp)      ## inverse temperature ##
D_cut = D                   ## bond dimension ##
Ns = int(2**((Niters)))     ## number of spatial lattice sites ##
Nt = Ns                     ## number of temporal lattice sites ##
vol = Ns**2                 ## lattice volume ##
numlevels = Niters          ## number of iterations ##
mcut = 50                   ## range for the order of modified bessel functions ##
Dn = int(D/2.0)             ## half of bond dimension ##

## function definitions ##
def SVD(t, l, r, D):
    """
    perfroms singular value decomposition for a tensor
    and returns U out of U, s, V
    t -> input tensor
    l -> list for left indices of tensor t 
    r -> list for right indices of tensor t
    """
    ## permute the input tensor ##
    t = permute(t, tuple(l+r)) if use_cuda else transpose(t, l+r)

    ## find number of elements of tensor in left and right direction ##
    left_index = [t.shape[i] for i in range(len(l))]
    right_index = [t.shape[i] for i in range(len(l), len(l) + len(r))]
    xsize, ysize = prod(left_index), prod(right_index)

    ## reshape tensor and do svd of it ##
    t = Treshape(t, (xsize, ysize)) if use_cuda else reshape(t, (xsize, ysize))
    u, _, _ = Tsvd(t, full_matrices=False) if use_cuda else svd(t, full_matrices=False)

    ## calculate the return variables ##
    size = u.shape[1]
    min(size, D)
    u = u[:,:D]
    u = Treshape(u,tuple(left_index+[D])) if use_cuda else reshape(u, left_index + [D]) 

    return u

def coarse_graining(t, ti, ti2):
    """
    performs coarse graining of tensor network and
    returns tensor, impure tensor and max(tensor)
    t -> pure tensor
    ti -> impurity tensor (used to compute magnetizations)
    norm -> maximum element of pure tensor t
    """
    if use_cuda:
        ## step - 1 ##
        AAdag = einsum('jabe,iecd,labf,kfcd->ijkl', t, t, t, t)
        U = SVD(AAdag,[0,1],[2,3],D_cut)
        A = einsum('abi,bjdc,acel,edk->ijkl', U, t, t, U)
        B = einsum('abi,bjdc,acel,edk->ijkl', U, ti, t, U)
        C = einsum('abi,bjdc,acel,edk->ijkl', U, ti2, t, U)

        ## step - 2 ##
        AAdag = einsum('aibc,bjde,akfc,flde->ijkl', A,A,A,A)
        U = SVD(AAdag,[0,1],[2,3],D_cut)
        AA = einsum('abj,iacd,cbke,del->ijkl', U, A, A, U)
        BA = einsum('abj,iadc,dbke,cel->ijkl', U, B, A, U)
        CA = einsum('abj,iadc,dbke,cel->ijkl', U, C, A, U)

        ## return varaibles ##
        norm = Tmax(AA) if use_cuda else max(AA)
    
    else:
        ## step - 1 ##
        AAdag = contract('jabe,iecd,labf,kfcd->ijkl', t, t, t, t)
        U = SVD(AAdag,[0,1],[2,3],D_cut)
        A = contract('abi,bjdc,acel,edk->ijkl', U, t, t, U)
        B = contract('abi,bjdc,acel,edk->ijkl', U, ti, t, U)
        C = contract('abi,bjdc,acel,edk->ijkl', U, ti2, t, U)

        ## step - 2 ##
        AAdag = contract('aibc,bjde,akfc,flde->ijkl', A,A,A,A)
        U = SVD(AAdag,[0,1],[2,3],D_cut)
        AA = contract('abj,iacd,cbke,del->ijkl', U, A, A, U)
        BA = contract('abj,iadc,dbke,cel->ijkl', U, B, A, U)
        CA = contract('abj,iadc,dbke,cel->ijkl', U, C, A, U)

        ## return varaibles ##
        norm = max(AA)

    ## normalize the tensors ##
    AA = AA/norm
    BA = BA/norm
    CA = CA/norm

    return AA, BA, CA, norm

def bessel_function(index, beta, delta):
    """
    returns sum of modified bessel functions of first kind
    index -> index of the initial tensor
    beta -> inverse temperature
    delta -> deformation parameter
    """
    val = [(iv(index - 2.0*i, beta*delta) * iv(i, beta*(1.0 - delta))) for i in range(-mcut, mcut+1)]
    return sum(val)

def init_tensor():
    """
    initializes the tensors for the simulation
    and returns the site tensor and impurity tensor
    corresponding to h and h1 external magentic field
    """
    print("Initializing site and magnetization tensors")
    start = time()
    ## compute weights for each index ##
    L = [sqrt(bessel_function(i, beta, delta)) for i in range(-Dn, Dn+1)]

    ## initialize the tensor using weights computed above ##
    if use_cuda:
        t1 = tensor(L)
        t2 = einsum('i,j,k,l->ijkl', t1,t1,t1,t1)
        t2_h = einsum('i,j,k,l->ijkl', t1,t1,t1,t1)
        t2_h1 = einsum('i,j,k,l->ijkl', t1,t1,t1,t1)
    else:
        t2 = contract('i,j,k,l->ijkl', L,L,L,L)
        t2_h = contract('i,j,k,l->ijkl', L,L,L,L)
        t2_h1 = contract('i,j,k,l->ijkl', L,L,L,L)

    ## multiply the tensors with other terms w.r.t derivation in the paper ##
    iv_cache_h = {index: iv(index, beta*h) for index in range(-mcut, mcut+1)}
    iv_cache_h1 = {index: iv(index, beta*h1) for index in range(-mcut, mcut+1)}

    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    val = 0.0
                    val_h = 0.0
                    val_h1 = 0.0
                    for s in range(-mcut,mcut+1):
                        index = (l + u - r - d + (2*s))
                        iv_h = iv_cache_h.get(index)
                        iv_h1 = iv_cache_h1.get(s)
                        if iv_h is None or iv_h1 is None:
                            iv_h = iv(index, beta*h)
                            iv_h1 = iv(s, beta*h1)
                            iv_cache_h[index] = iv_h
                            iv_cache_h1[s] = iv_h1
                        val += iv_h * iv_h1
                        val_h += 0.5 * (iv(index-1, beta*h) + iv(index + 1, beta*h)) * iv_h1
                        val_h1 += 0.5 * iv_h * (iv(s-1, beta*h1) + iv(s+1, beta*h1))

                    t2[l+Dn][r+Dn][u+Dn][d+Dn] *= val
                    t2_h[l+Dn][r+Dn][u+Dn][d+Dn] *= val_h
                    t2_h1[l+Dn][r+Dn][u+Dn][d+Dn] *= val_h1
    end = time()
    print("Finished initializing tensors in {} hours".format(round((end-start)/3600.0,5)))
    return t2, t2_h, t2_h1, end-start

## main driver function of the program ##
if __name__ == "__main__":
    start = time()
    print("\nSTART : {}".format(ctime()))
    
    ## initialize the tensors ##
    T, Tim, Tim2, tinit_time = init_tensor()
    
    ## normalize these tensors ##
    norm = Tmax(T) if use_cuda else max(T)
    T = T/norm
    Tim = Tim/norm
    Tim2 = Tim2/norm

    ## compute the initial partition function ##
    if use_cuda:
        Z = einsum('geca,cfgb,hade,dbhf -> ''', T, T, T, T)  
    else:
        Z = contract('geca,cfgb,hade,dbhf -> ''', T, T, T, T)
    
    ## exectue the HOTRG algorithm ##
    N = 1
    C = log(norm)

    ## HOTRG method loop ##
    print("\nExectuing HOTRG Algorithm\n-------------------------")
    for i in range(Niters):
        current_time = ctime()
        start = time()
        T, Tim, Tim2, norm = coarse_graining(T, Tim, Tim2)

        ## update scalar parameters ##
        C = log(norm) + 4.0*C
        N *= 4.0
        end = time()
        delta_t = round(((end-start)/60.0),5)
        print("Iteration = {}/{} started at {}, took {} minutes".format(i+1, Niters, current_time, delta_t))

        ## printing data ##
        if i == Niters-1:
            print("-------------------------\nFinshed HOTRG Algorithm\n")
            print("Printing results now\n-------------------------")

            ## calculation of the phases ##
            NUM = einsum('ruru -> ''',T)**2 if use_cuda else contract('ruru -> ''',T)**2
            Y1 = NUM/einsum('rulu, ldrd -> ''',T, T) if use_cuda else contract('rulu, ldrd -> ''',T, T)
            X1 = NUM/einsum('ruld, ldru -> ''',T, T) if use_cuda else contract('ruld, ldru -> ''',T, T)
            print ("X = {}\nY = {}".format(X1, Y1))

            ## update the partition function ##
            Z1 = einsum('aibj,bkal->ijkl', T, T) if use_cuda else contract('aibj,bkal->ijkl', T, T)
            Z = einsum('abcd,badc -> ''', Z1, Z1) if use_cuda else contract('abcd,badc -> ''', Z1, Z1)

            ## update the free energy ##
            Free = -Temp * (log(Z) + 4.0*C) / (4.0*N)

            ## calculate the magnetization ##
            P = einsum('aibj,bkal->ijkl', Tim, T) if use_cuda else contract('aibj,bkal->ijkl', Tim, T)
            P2 = einsum('aibj,bkal->ijkl', Tim2, T) if use_cuda else contract('aibj,bkal->ijkl', Tim2, T)
            P = einsum('abcd,badc -> ''', P, Z1) if use_cuda else contract('abcd,badc -> ''', P, Z1)
            P2 = einsum('abcd,badc -> ''', P2, Z1) if use_cuda else contract('abcd,badc -> ''', P2, Z1)  
            mag = (P/Z)
            mag2 = (P2/Z)
        
    ## print the data to and output file ##
    hfield = h1 if h1 != 0.0 else h
    f=open("h_{}.txt".format(hfield), "a+")    
    f.write("""%4.10f \t %4.10f \t %4.10f \t %4.10f \t %2.0f \t%2.0f \t %2.3e \t %2.3e \t %2.4f \n""" %(Temp, Free, mag, mag2, Niters, D_cut, h, h1, delta)) 
    f.close()

    ## print the data to console ##         
    print("Temperature = %4.10f \nFree Energy = %4.10f \nMagnetization 1 = %4.10f"%(Temp,Free,mag))  
    print("Magnetization 2 = %4.10f \nNumber of Iterations = %2.0f \nBond Dimension = %2.0f"%(mag2,Niters,D_cut))
    print("Magnetic Field 1 = %2.3e \nMagnetic Field 2 = %2.3e \nDelta = %2.4f"%(h,h1,delta))

    end = time()
    print("-------------------------")
    print("\nEND : {}\n".format(ctime()))
    print("Runtime = {} hours".format(round((end-start+tinit_time)/3600.0,5)))

#####################################
###### 2d Generalized XY Model ######
#####################################