import itertools
import functools
import numpy as np
import tnad.FeatureMap as fm
import tnad.embeddings as e
from tnad.losses import loss_miss, loss_reg, LossWrapper
from tnad.gradients import gradient_miss, gradient_reg
import math
import quimb.tensor as qtn
from tqdm import tqdm
import autoray as a
from jax import grad
from jax import numpy as jnp

def local_update_sweep_dyncanonization_renorm(P, n_epochs, n_iters, data, batch_size, alpha, lamda_init, bond_dim, decay_rate=None, expdecay_tol=None):
    N_features = P.nsites

    loss_array = []
    for epoch in range(n_epochs):
        for it in (pbar := tqdm(range(n_iters))):        
            pbar.set_description("Epoch #"+str(epoch)+", sample in batch:")
            # define sweeps
            sweeps = itertools.chain(zip(list(range(0,N_features-1)), list(range(1,N_features))), reversed(list(zip(list(range(1,N_features)),list(range(0,N_features-1))))))
            for sweep_it, sites in enumerate(sweeps):
                [sitel, siter] = sites
                site_tags = [P.site_tag(site) for site in sites]
                # canonize P with root in sites
                ortog_center = sites
                P.canonize(sites, cur_orthog=ortog_center)
                # copy P as reference
                P_ref = P.copy(deep=True)
                # pop site tensor
                [origl, origr] = P.select_tensors(site_tags, which="any")
                tensor_orig = origl & origr ^ all
                # memorize bond between 2 selected sites
                bond_ind_removed = P.bond(site_tags[0], site_tags[1])

                #virtual bonds
                #    left
                if sitel == 0 or (sitel == N_features-1 and sitel>siter): vindl = []
                elif sitel>0 and sitel<siter: vindl = [P.bond(sitel-1, sitel)]
                else: vindl = [P.bond(sitel, sitel+1)]
                #    right
                if siter == N_features - 1 or (siter == 0 and siter<sitel): vindr = []
                elif siter < N_features-1 and siter>sitel: vindr = [P.bond(siter, siter+1)]
                else: vindr = [P.bond(siter-1, siter)]

                # remove site tags of poped sites
                P.delete(site_tags, which="any")

                grad_miss=0; loss_miss_batch=0
                for sample in data[it]:
                    # create MPS for input sample
                    phi, _ = fm.embed(sample.flatten(), fm.trigonometric)
                    
                    #calculate loss
                    loss_miss_batch += loss_miss(phi, P_ref)
                    
                    #calculate gradient
                    grad_miss += gradient_miss(phi, P_ref, P, sites)
                # total loss
                loss = (1/batch_size)*(loss_miss_batch)
                loss_array.append(loss)

                # gradient of loss miss
                grad_miss.drop_tags()
                grad_miss.add_tag(site_tags[0]); grad_miss.add_tag(site_tags[1])
                # gradient of loss reg
                # grad_regular = gradient_reg(P_ref, P, alpha, sites, N_features)
                # if grad_regular != 0:
                #     grad_regular.drop_tags()
                #     grad_regular.add_tag(site_tags[0]); grad_regular.add_tag(site_tags[1])
                # total gradient
                total_grad = (1/batch_size)*grad_miss

                # update tensor
                if epoch > expdecay_tol:
                    if decay_rate != None:
                        # exp. decay of lamda
                        lamda = lamda_init*math.pow((1 - decay_rate/100),epoch)
                        tensor_new = tensor_orig - lamda*total_grad
                else:
                    tensor_new = tensor_orig - lamda_init*total_grad

                # normalize updated tensor
                tensor_new.normalize(inplace=True)

                # split updated tensor in 2 tensors
                lower_ind = [f'b{sitel}'] if f'b{sitel}' in P.lower_inds else []
                [tensorl, tensorr] = tensor_new.split(get="tensors", left_inds=[*vindl, P.upper_ind(sitel), *lower_ind], bond_ind=bond_ind_removed, max_bond=bond_dim)

                # link new tensors to P back
                for site, tensor in zip(sites, [tensorl, tensorr]):
                    tensor.drop_tags()
                    tensor.add_tag(P.site_tag(site))
                    P.add_tensor(tensor)
    return P, loss_array

def get_sample_grad(sample, embed_func, P, P_rem, tensor):
    # create MPS for input sample
    phi, _ = fm.embed(sample.flatten(), embed_func)

    #calculate gradient
    grad_miss = gradient_miss(phi, P, P_rem, [tensor])
    return grad_miss

def get_sample_loss(sample, embed_func, P):
    # create MPS for input sample
    phi, _ = fm.embed(sample.flatten(), embed_func)

    #calculate loss
    loss_miss_batch = loss_miss(phi, P)
    return loss_miss_batch

def get_total_grad(P, tensor, data, embed_func, batch_size, alpha):
    P_rem = P.copy(deep=True)
    
    site_tag = P_rem.site_tag(tensor)
    # remove site tag of poped sites
    P_rem.delete(site_tag, which="any")

    # paralelize
    grad_miss = []
    for i, sample in enumerate(data):
        output_per_sample = get_sample_grad(sample, embed_func, P, P_rem, tensor)
        grad_miss.append(output_per_sample)
    
    # gradient of loss miss
    grad_miss = sum(grad_miss)
    grad_miss.drop_tags()
    grad_miss.add_tag(site_tag)
    # gradient of loss reg
    grad_regular = gradient_reg(P, P_rem, alpha, [tensor])
    if grad_regular != 0:
        grad_regular.drop_tags()
        grad_regular.add_tag(site_tag)
    # total gradient
    total_grad = (1/batch_size)*(grad_miss) + grad_regular
    return total_grad

def global_update_costfuncnorm(P, n_epochs, n_iters, data, batch_size, alpha, lamda_init, lamda_init_2, bond_dim, decay_rate=None, expdecay_tol=None):
    loss_array = []
    n_tensors = P.nsites
    for epoch in range(n_epochs):
        for it in (pbar := tqdm(range(n_iters))):        
            pbar.set_description("Epoch #"+str(epoch)+", sample in batch:")
            # paralelize
            grad_per_tensor=[]
            for tensor in range(n_tensors):
                embed_func = fm.trigonometric
                output_per_tensor = get_total_grad(P, tensor, data[it], embed_func, batch_size, alpha) # get grad per tensor
                grad_per_tensor.append(output_per_tensor)
            
            # get loss per sample
            loss_value = 0
            for i, sample in enumerate(data[it]):
                embed_func = fm.trigonometric
                output_per_sample = get_sample_loss(sample, embed_func, P)
                loss_value += output_per_sample
            # get total loss
            reg = loss_reg(P, alpha)
            total_loss = (1/batch_size)*(loss_value) + reg
            loss_array.append([total_loss,(1/batch_size)*(loss_value),reg])


            # print(f'grad norm: {grad_tn.norm()}')
            print(f'grad norm per tensor: {[tensor.norm() for tensor in grad_per_tensor]}')
            print(f'P norm: {P.H@P}')

            # update P
            # no need to paralelize
            for tensor in range(n_tensors):
                site_tag = P.site_tag(tensor)
                (tensor_orig,) = P.select_tensors(site_tag, which="any")
                
                if epoch >= expdecay_tol:
                    if decay_rate != None:
                        # exp. decay of lamda
                        if epoch == expdecay_tol: lamda = lamda_init_2
                        else: lamda = lamda_init_2*math.pow((1 - decay_rate/100),epoch)
                        tensor_orig.modify(data = tensor_orig.data - lamda*grad_per_tensor[tensor].transpose_like(tensor_orig).data)
                else:
                    tensor_orig.modify(data = tensor_orig.data - lamda_init*grad_per_tensor[tensor].transpose_like(tensor_orig).data)
                
    return P, loss_array

# To use this with bADAM all_phis must be formated as a 2D array of [n_batches][batch_size]
# everytime bADAM calls this loss function it will fix which batch to use (every iteration will use the next one)
def get_losses(all_phis,P,batch=None,alpha=0.4,total=True):
    loss_value = 0
    if batch==None:
        phis = all_phis 
    else: 
        phis = all_phis[batch%len(all_phis)]
    batch_size = len(phis)
    for phi in phis:
        loss_value += loss_miss(phi, P, 1/batch_size)

    if total:
        return loss_value + loss_reg(P, alpha)
    else:
        return loss_value,loss_reg(P, alpha)

def automatic_differentiation(P, n_epochs, n_iters, data, batch_size, alpha, lamda_init, lamda_init_2, bond_dim, decay_rate=None, expdecay_tol=None, alg_depth=2, jit_fn=False, par_client=None, loss_detail=False, backend = 'jax', optimizer='L-BFGS-B'):

    loss_array = []
    if backend == 'jax':
        from jax.config import config
        # to fix nans in jax, we need this setting
        config.update("jax_enable_x64", True)
    
    for epoch in range(n_epochs):
        for it in (pbar := tqdm(range(n_iters))):
            pbar.set_description("Epoch #"+str(epoch)+", sample in batch:")

            # This will make it parallelizable as all these components for the loss function will be computed separately
            # embed_func = fm.trigonometric
            # phis = [fm.embed(sample.flatten(), embed_func)[0] for sample in data[it]]
            phis = [e.embed(sample.flatten(), e.trigonometric(k=1)) for sample in data[it]]
            
            # if optimizer=='ADAM':
            #     alg_depth = 0

            if alg_depth==3:
                loss_fn = functools.partial(get_losses,phis,alpha=alpha,total=True)
                wrapper = LossWrapper(loss_fn, P)
                get_grad = grad(wrapper)
            else:
                # Parallelize (if we have mulptiple loss_fns as described above)
                loss_fns = [
                    functools.partial(loss_miss, phi, coeff=(1/batch_size))
                    for phi in phis
                ] + [ functools.partial(loss_reg, alpha=alpha, backend=backend) ]

                if backend=='jax':
                    tnopt = qtn.TNOptimizer(
                        P,
                        loss_fn=loss_fns,
                        # loss_constants={'phi':phis[0]}, # In loss constants we can specify which parameters we do not want to differentiate over. In this case we don't need to put samples because we fixed it with "partial" above
                        # loss_kwargs={'coeff':(1/batch_size)}, 
                        autodiff_backend=backend,
                        optimizer=optimizer, # here we can use any method in scipy.minimize or in quimb ('sgd','rmsprop','adam','nadam')
                        jit_fn = jit_fn,
                        executor=par_client,
                    )
                else:
                    tnopt = qtn.TNOptimizer(P,loss_fn=loss_fns,autodiff_backend=backend,optimizer=optimizer,executor=par_client)

            if alg_depth==0:
                P = tnopt.optimize(1)
                lmiss, lreg = get_losses(phis,P,alpha=alpha,total=False)
                loss_array.append(tnopt.res.fun)
                if loss_detail:
                    loss_array[-1]=(loss_array[-1],lmiss,lreg)
            elif alg_depth==1:
                x = tnopt.vectorizer.vector  # P is already stored in the appropriate vector form when initializing tnopt
                # arrays = tnopt.vectorizer.unpack()
                # print(f'x is = {x}')
                # print(len(x))
                loss, grad_full = tnopt.vectorized_value_and_grad(x) # extract the loss and the gradient
                loss_array.append(loss)
                print(f'grad is {grad_full}')
                tnopt.vectorizer.vector[:] = grad_full
                grad_tn = tnopt.get_tn_opt()
                # print(f'grad_tn is {grad_tn}')
                # print(f'grad norm: {grad_tn.norm()}')
                # print(f'grad norm per tensor: {[tensor.norm() for tensor in grad_tn]}')
                # print(f'P norm: {P.H@P}')

                if loss_detail:
                    loss_array[-1]=(loss,)+get_losses(phis,P,alpha=alpha,total=False)

                for tensor in range(P.nsites):
                    site_tag = P.site_tag(tensor)
                    tensor_orig = P.select_tensors(site_tag, which="any")[0]
                    if epoch >= expdecay_tol:
                        if decay_rate != None:
                            # exp. decay of lamda
                            if epoch == expdecay_tol: lamda = lamda_init_2
                            else: lamda = lamda_init_2*math.pow((1 - decay_rate/100),epoch)
                            tensor_orig.modify(data = tensor_orig.data - lamda*grad_tn[tensor].transpose_like(tensor_orig).data)
                    else:
                        tensor_orig.modify(data = tensor_orig.data - lamda_init*grad_tn[tensor].transpose_like(tensor_orig).data)

            elif alg_depth==2:
                if it==0:
                    x = tnopt.vectorizer.vector  # P is already stored in the appropriate vector form when initializing tnopt #####Should be equivalent to "x = tnopt.vectorizer.pack(P.arrays)"
                loss, grad_full = tnopt.vectorized_value_and_grad(x) # extract the loss and the gradient
                loss_array.append(loss)
                if epoch > expdecay_tol:
                    if decay_rate != None:
                        # exp. decay of lamda
                        lamda = lamda_init*math.pow((1 - decay_rate/100),epoch)
                        x = x - lamda*grad_full
                else:
                    x = x - lamda_init*grad_full
            # using a wrapper to avoid tnopt altogether
            elif alg_depth==3: 
                arrays = tuple(map(jnp.asarray, P.arrays))
                grad_arrays = list(map(a.to_numpy, get_grad(arrays)))

                if epoch > expdecay_tol:
                    if decay_rate != None:
                        lamda = lamda_init*math.pow((1 - decay_rate/100),epoch)
                else:
                    lamda = lamda_init

                for tensor, grad_array in zip(P.tensors, grad_arrays):
                    tensor.modify(data=tensor.data - lamda * grad_array)

                lmiss,lreg = get_losses(phis,P,alpha=alpha,total=False)
                loss_array.append((lmiss+lreg,lmiss,lreg))
            else:
                print('Something went wrong. I skipped all possible methods!')
        
    if alg_depth==2:
        print('Remember that with alg_depth=2 the separate contribution of loss_reg and loss_miss cannot be calculated at each update')
        tnopt.vectorizer.vector[:] = x
        P = tnopt.get_tn_opt()

    return P, loss_array

####At the moment this is not necessary as it is conecptually replaced by alg_depth=3####
# I left it here to see if some of this makes sense to put in place after the refactor
# Work in progress 
# from scipy.optimize import minimize
# def optimize(tnopt,iter,tol=None,method=None,**options):

#     fun = tnopt.vectorized_value_and_grad

#     if method==None:
#         method = tnopt._method

#     try:
#         tnopt._maybe_init_pbar(n)
#         res = minimize(
#             fun=fun,
#             jac=True,
#             hessp=None,
#             x0=tnopt.vectorizer.vector,
#             tol=tol,
#             bounds=tnopt.bounds,
#             method=method,
#             options=dict(maxiter=iter, **options),
#         )
#         tnopt.vectorizer.vector[:] = res.x
#     except KeyboardInterrupt:
#         pass
#     finally:
#         tnopt._maybe_close_pbar()

#     return tnopt.get_tn_opt(), res.fun



# In addition to adapting the class, it is important to think how we make it reach JAX. Based on QUIMB, the following is how you can do it:
# tnopt = qtn.TNOptimizer( ... , optimizer = bADAM , ... )
# if the argument in optimizer is an instance of the class it won't work! It has to be the class object.

##### This on the other hand could be interesting to develop #####
# Work in progress
class bADAM: # adaptation of adam with batches. This should serve as an example of how to adapt other strategies.
    """Stateful ``scipy.optimize.minimize`` compatible implementation of
    ADAM - http://arxiv.org/pdf/1412.6980.pdf.

    Adapted from ``autograd/misc/optimizers.py``.
    """

    def __init__(self):
        from scipy.optimize import OptimizeResult
        self.OptimizeResult = OptimizeResult
        self._i = 0
        self._m = None
        self._v = None

    def get_m(self, x):
        if self._m is None:
            self._m = np.zeros_like(x)
        return self._m

    def get_v(self, x):
        if self._v is None:
            self._v = np.zeros_like(x)
        return self._v

    def __call__(self, fun, x0, jac, args=(), learning_rate=0.001, beta1=0.9,
                 beta2=0.999, eps=1e-8, maxiter=1000, callback=None,
                 bounds=None, **kwargs):
        x = x0
        m = self.get_m(x)
        v = self.get_v(x)

        for _ in range(maxiter):
            self._i += 1

            # jac is called with argument i to indicate which batch to use. 
            # We iterate over all of them, and for a given one, the update is done using all samples in the batch at once
            g = jac(x,self._i)

            if callback and callback(x):
                break

            m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(self._i))  # bias correction.
            vhat = v / (1 - beta2**(self._i))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

            if bounds is not None:
                x = np.clip(x, bounds[:, 0], bounds[:, 1])

        # save for restart
        self._m = m
        self._v = v

        return self.OptimizeResult(
            x=x, fun=fun(x), jac=g, nit=self._i, nfev=self._i, success=True)