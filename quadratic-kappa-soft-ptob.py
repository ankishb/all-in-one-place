def soft_kappa_grad_hess(y, p):
    '''
        Returns first and second derivatives of the objective with respect to predictions `p`. 
        `y` is a vector of corresponding target labels.  
    '''
    norm = p.dot(p) + y.dot(y)
    
    grad = -2 * y / norm + 4 * p * np.dot(y, p) / (norm ** 2)
    hess = 8 * p * y / (norm ** 2) + 4 * np.dot(y, p) / (norm ** 2)  - (16 * p ** 2 * np.dot(y, p)) / (norm ** 3)
    return grad, hess

def soft_kappa(preds, dtrain):
    '''
        Having predictions `preds` and targets `dtrain.get_label()` this function coumputes soft kappa loss.
        NOTE, that it assumes `mean(target) = 0`.
        
    '''
    target = dtrain.get_label()
    return 'kappa' ,  -2 * target.dot(preds) / (target.dot(target) + preds.dot(preds))
