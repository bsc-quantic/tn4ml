from tnad.embeddings import embed
from tnad.gradients import gradient_miss, gradient_reg

def get_total_grad(P, tensor, data, loss, batch_size=None, alpha=0):
    """ data = array of MPSs """
    history = dict()
    
    P_rem = P.copy(deep=True)
    
    site_tag = P_rem.site_tag(tensor)
    # remove site tag of poped sites
    P_rem.delete(site_tag, which="any")

    # paralelize
    grad_miss = []
    for i, phi in enumerate(data):
        # grad per sample
        grad_miss.append(gradient_miss(phi, P, P_rem, [tensor]))
   
    # gradient of error part
    grad_miss = sum(grad_miss)
    grad_miss.drop_tags()
    grad_miss.add_tag(site_tag)
    # gradient of regularization part
    grad_regular = gradient_reg(P, P_rem, alpha, [tensor])
    if grad_regular != 0:
        grad_regular.drop_tags()
        grad_regular.add_tag(site_tag)
    # total gradient
    history['grad_miss'] = (1/batch_size)*(grad_miss)
    history['grad_reg'] = grad_regular
    total_grad = (1/batch_size)*(grad_miss) + grad_regular
    
    return total_grad, history

def get_total_loss(P, data, loss, batch_size=None, alpha=0):
    """ data = array of MPSs """
    history = dict()
    
    # paralelize
    loss_value=0
    for i, phi in enumerate(data):
        # loss per sample
        loss_value += loss['error'](P, phi)
   
    # total loss
    history['loss_miss'] = (1/batch_size)*(loss_value)
    history['loss_reg'] = loss['reg'](P)
    
    total_loss = (1/batch_size)*(loss_value) + alpha*loss['reg'](P)
    
    return total_loss, history