import abc
from quimb import tensor as qtn
from tqdm import auto as tqdm
import funcy


def direct_gradient_descent(tensor, grad, lambd=0.01):
    return tensor - lambd * grad


# TODO add hyperparameters arguments
class Model:
    def fit_step(self, loss_fn, strategy="dmrg", optimizer=direct_gradient_descent, niter=1, **kwargs):
        if isinstance(strategy, Strategy):
            pass
        elif strategy in ["sweeps", "local", "dmrg"]:
            strategy = Sweeps()
        elif strategy == ["global"]:
            strategy = Global()
        else:
            raise ValueError(f'Strategy "{strategy}" not found')

        for sites in strategy.iterate_sites(self.sites):
            # contract tensors (if needed)
            strategy.prehook(self, sites)

            target_site_tags = tuple(self.site_tag(site) for site in funcy.flatten(sites))
            opt = qtn.TNOptimizer(
                self,
                loss_fn=loss_fn,
                optimizer=optimizer,
                tags=target_site_tags,
                **kwargs,
            )

            if isinstance(optimizer, str):
                psi = opt.optimize(niter)
            else:
                x = opt.vectorizer.vector
                _, grads = opt.vectorized_value_and_grad(x)
                grads = opt.vectorizer.unpack(grads)

                tensors = psi.select_tensors(target_site_tags, which="any")
                for tensor, grad in zip(tensors, grads):
                    tensor.modify(data=optimizer(tensor.data, grad))

            # split tensors (if needed) & renormalize (if configured)
            strategy.posthook(self, sites)

    @abc.abstractmethod
    def score(self, x):
        pass


class Strategy:
    """Decides how the gradients are computed. i.e. compute the gradients of each tensor separately or only of one site."""

    def __init__(self, renormalize=False):
        self.renormalize = renormalize

    def prehook(self, model, sites):
        """Modify `model` before computing gradient(s). Usually contract tensors."""
        pass

    def posthook(self, model, sites):
        """Modify `model` after optimizing tensors. Usually split tensors."""
        pass

    @abc.abstractmethod
    def iterate_sites(self, sites):
        pass

    @abc.abstractmethod
    def posthook(self, psi):
        pass


from .smpo import SpacedMatrixProductOperator
