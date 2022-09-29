from autoray import do

def loss_miss(phi,P,coeff=1):
    half_tn = P&phi
    loss = do('power',do('log',half_tn.H@(half_tn)) - 1, 2)
    return coeff * loss

def loss_reg(P,alpha=0.4,backend=None):
    return alpha*do('maximum',do('log',P.H@P),0)