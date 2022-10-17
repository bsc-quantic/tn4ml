from autoray import do

def loss_miss(phi,P,coeff=1):
    half_tn = P&phi
    loss = do('power',do('log',half_tn.H@(half_tn)) - 1, 2)
    return coeff * loss

def loss_reg(P,alpha=0.4,backend=None):
    return alpha*do('maximum',do('log',P.H@P),0)

class LossWrapper:
	def __init__(self, loss_fn, tn):
		self.loss_fn = loss_fn
		self.tn = tn

	def __call__(self, arrays):
		tn = self.tn.copy()

		for tensor, array in zip(tn.tensors, arrays):
			tensor.modify(data=array)

		return self.loss_fn(tn)