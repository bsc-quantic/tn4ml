from tnad.embeddings import embed
from tnad.gradients import gradient_miss, gradient_reg

def get_total_grad_and_loss(P, tensor, data, loss, batch_size=None, alpha=0):
    """ data = array of MPSs """
    
    P_rem = P.copy(deep=True)
    
    site_tag = P_rem.site_tag(tensor)
    # remove site tag of poped sites
    P_rem.delete(site_tag, which="any")

    # paralelize
    grad_miss = []; loss_value=0
    for i, phi in enumerate(data):
        # grad per sample
        grad_miss.append(gradient_miss(phi, P, P_rem, [tensor]))
        # loss per sample
        loss_value += loss['error'](P, phi)
   
    # total loss
    total_loss = (1/batch_size)*(loss_value) + alpha*loss['reg'](P)
    
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
    total_grad = (1/batch_size)*(grad_miss) + grad_regular
    
    return total_grad, total_loss