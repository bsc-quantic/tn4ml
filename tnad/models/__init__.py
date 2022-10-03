import abc
from quimb import tensor as qtn
from tqdm import auto as tqdm


def direct_gradient_descent(tensor, grad, lambd=0.01):
    return tensor - lambd * grad


class Model:
    def fit(self, loss_fn, strategy="local", optimizer=None, update_fn=direct_gradient_descent, renormalize=False, max_it=100, autodiff_backend="jax", **autodiff_kwargs):
        if isinstance(strategy, Strategy):
            pass
        elif strategy in ["sweeps", "local"]:
            strategy = Sweeps
        elif strategy == "global":
            strategy = Global
        else:
            raise ValueError(f'Strategy "{strategy}" not found')

        with tqdm(range(max_it)) as progressbar:
            progressbar.set_postfix(loss=f"")

            for it in progressbar:
                # TODO
                strategy.preprocess(...)

                optkwargs = {}
                if optimizer is not None:
                    optkwargs["optimizer"] = optimizer

                opt = qtn.TNOptimizer(self, loss_fn=loss_fn, loss_constants={}, autodiff_backend=autodiff_backend, progbar=False, **optkwargs)

                if optimizer is not None:
                    psi = opt.optimize(1)
                else:
                    x = opt.vectorizer.vector
                    _, grads = opt.vectorized_value_and_grad(x)
                    grads = opt.vectorizer.unpack(grads)

                    for tensor, grad in zip(psi.tensors, grads):
                        tensor.modify(data=update_fn(tensor.data, grad))

                # TODO
                strategy.postprocess(...)

    @abc.abstractmethod
    def score(self, x):
        pass


# # TODO what should the `Strategy` do?
# can we compute partial derivatives with jax? that way we could use jax inside the `Sweeps` and `Global` strategies and strategies would just pre-contract and ask jax for grads
# maybe for manual diff can we put new rules for our models?
class Strategy:
    """Tells us how the gradients are computed and when have to be applied."""

    @abc.abstractmethod
    def preprocess(self, psi):
        pass

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def postprocess(self, psi):
        pass


from .smpo import SpacedMatrixProductOperator
