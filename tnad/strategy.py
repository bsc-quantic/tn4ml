import abc


class Strategy:
    """
    Decides how the gradients are computed. i.e. computes the gradients of each tensor separately or only of one site.
    
    Parameters
        renormalize: Flag for renormalization. `bool`, default=False.
    """

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
    """
    DMRG-like local optimization.
    
    Parameters
        grouping: Number of tensors to group together. `int`, default=2.
        two_way: Flag indicating wheather sweeping happens two-way or one-way. `bool`, default=True (two-way sweep).
        split_opts: Additional args passed to `model.split_tensor()`.
    """

    def __init__(self, grouping: int = 2, two_way=True, split_opts={"cutoff": 1e-3}, **kwargs):
        self.grouping = grouping
        self.two_way = two_way
        self.split_opts = split_opts
        super().__init__(**kwargs)

    def iterate_sites(self, model):
        for i in model.sites[: len(model.sites) - self.grouping + 1]:
            yield tuple(model.sites[i + j] for j in range(self.grouping))

    def prehook(self, model, sites):
        model.canonize(sites)

        sitetags = tuple(model.site_tag(site) for site in sites)
        model.contract_tags(sitetags, inplace=True)

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
        site_ind_prefix = model.upper_ind_id.rstrip("{}")

        vindl = [model.upper_ind(sitel)] + ([model.bond(sitel - 1, sitel)] if sitel > 0 else [])
        # vindr = [model.upper_ind(siter)] + ([model.bond(siter, siter + 1)] if siter < model.nsites - 1 else []) + ([model.lower_ind(siter)] if model.tensors[siter].ndim == 4 or (siter==model.L-1 and model.tensors[siter].ndim == 3) else [])

        lower_ind = [f"{model.lower_ind_id}{sitel}"] if f"{model.lower_ind_id}{sitel}" in model.lower_inds else []
        model.split_tensor(sitetags, left_inds=[*vindl, *lower_ind], **self.split_opts)
        # fix tags
        for tag in sitetags:
            for tensor in model.select_tensors(tag):
                tensor.drop_tags()
                site_ind = next(filter(lambda ind: ind.removeprefix(site_ind_prefix).isdecimal(), tensor.inds))
                site = site_ind.removeprefix(site_ind_prefix)
                tensor.add_tag(model.site_tag(site))


class Global(Strategy):
    """
    Global optimization through Gradient descent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def iterate_sites(self, model):
        yield model.sites

    def posthook(self, model, sites):
        # renormalize
        if self.renormalize:
            model.normalize(inplace=True)
