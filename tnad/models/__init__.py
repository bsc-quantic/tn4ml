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


class Sweeps(Strategy):
    """DMRG-like local optimization."""

    def __init__(self, grouping: int = 2, two_way=True, split_opts={"cutoff": 1e-3}, **kwargs):
        self.grouping = grouping
        self.two_way = two_way
        self.split_opts = split_opts
        super().__init__(**kwargs)

    def iterate_sites(self, sites):
        for i in sites[: len(sites) - self.grouping + 1]:
            yield tuple(sites[i + j] for j in range(self.grouping))

    def prehook(self, model, sites):
        model.canonize(sites)

        sitetags = tuple(model.site_tag(site) for site in sites)
        model.contract_tags(sitetags)

    def posthook(self, model, sites):
        sitetags = [model.site_tag(site) for site in sites]
        tensor = model.select_tensors(sitetags, which="all")[0]

        # normalize
        if self.renormalize:
            tensor.normalize(inplace=True)

        # split tensor
        # TODO right now only support grouping <= 2
        if self.grouping > 2:
            raise RuntimeError(f"{self.grouping=} > 2")

        sitel, siter = sites
        site_ind_prefix = model.site_ind_id.rstrip("{}")

        vindl = [model.site_ind(sitel)] + ([model.bond(sitel - 1, sitel)] if sitel > 0 else [])
        vindr = [model.site_ind(siter)] + ([model.bond(siter, siter + 1)] if siter < model.nsites - 1 else [])

        model.split_tensor(sitetags, left_inds=vindl, right_inds=vindr, **self.split_opts)

        # fix tags
        for tag in sitetags:
            for tensor in model.select_tensors(tag):
                tensor.drop_tags()
                site_ind = next(filter(lambda ind: ind.removeprefix(site_ind_prefix).isdecimal(), tensor.inds))
                site = site_ind.removeprefix(site_ind_prefix)
                tensor.add_tag(model.site_tag(site))



from .smpo import SpacedMatrixProductOperator
