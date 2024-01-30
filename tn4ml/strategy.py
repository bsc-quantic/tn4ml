import abc
import quimb.tensor as qtn

class Strategy:
    """Decides how the gradients are computed. i.e. computes the gradients of each tensor separately or only of one site.

    Attributes
    ----------
    renormalize : bool
        Flag for renormalization. *Default*=**False**.
    """

    def __init__(self, renormalize=False):
        self.renormalize = renormalize

    def prehook(self, model, sites):
        """Modify `model` before computing gradient(s). Usually contract tensors.

        Parameters
        ----------
        model : :class:`tn4ml.models.Model``
            Model
        sites : sequence of `str`
            List of tensors' tags.
        """
        pass

    def posthook(self, model, sites):
        """Modify `model` after optimizing tensors. Usually split tensors.

        Parameters
        ----------
        model : :class:`tn4ml.models.Model``
            Model
        sites : sequence of `str`
            List of tensors' tags.
        """
        pass

    @abc.abstractmethod
    def iterate_sites(self, sites):
        """ Function for iterating selected tensors.

        Parameters
        ----------
        sites : sequence of `str`
            List of tensors' tags.
        """
        pass

class Sweeps(Strategy):
    """DMRG-like local optimization.

    Attributes
    ----------
    grouping : int
    Number of tensors to group together. *Default*=**2**.
    two_way : bool
        Flag indicating wheather sweeping happens two-way or one-way. *Default*=**True** *(two-way sweep)*.
    split_opts: optional
        Additional args passed to ``model.split_tensor()``.
    """
    # TODO still not fully functional for grouping=2
    def __init__(self, grouping: int = 2, two_way=True, split_opts={"cutoff": 0.}, **kwargs):
        # TODO right now only support grouping <= 2
        if grouping > 2:
            raise ValueError(f"{grouping=} > 2")
        self.grouping = grouping
        self.two_way = two_way
        self.split_opts = split_opts
        super().__init__(**kwargs)

    def iterate_sites(self, model):
        if self.grouping == 1:
            sites = [site for site in model.sites]
            
            for site in list(reversed(model.sites[:-1])):
                sites.append(site)
            for site in sites:
                yield site
        else:
            for i in model.sites[: len(model.sites) - self.grouping + 1]:
                yield tuple(model.sites[i + j] for j in range(self.grouping))

    def prehook(self, model, sites):
        if self.grouping == 1:
            return
        
        model.canonize(sites)
        
        # remembed bond_dim
        #self.bond_dim_split = model.bond_size(sites[0], sites[1])
        sitetags = tuple(model.site_tag(site) for site in sites)
        model.contract_tags(sitetags, inplace=True)

    def posthook(self, model, sites):
        if self.grouping == 1:
            return
        sitetags = [model.site_tag(site) for site in sites]
        tensor = model.select_tensors(sitetags, which="all")[0]
        # normalize
        if self.renormalize:
            tensor.normalize(inplace=True)

        # split tensor
        sitel, siter = sites
        if isinstance(model, qtn.MatrixProductState):
            site_ind_prefix = model.site_ind_id.rstrip("{}")
            vindl = [f'{site_ind_prefix}{sitel}'] + ([model.bond(sitel - 1, sitel)] if sitel > 0 else [])
            vindr = [f'{site_ind_prefix}{siter}']
            qtn.tensor_core.tensor_split(model, left_inds=[*vindl], right_inds=[*vindr], max_bond=model.max_bond(), **self.split_opts)
            #model.split_tensor(sitetags, left_inds=[*vindl], max_bond=model.max_bond(), **self.split_opts)
        else:
            site_ind_prefix = model.upper_ind_id.rstrip("{}")
            vindr = [model.upper_ind(siter)] + ([model.bond(siter, siter + 1)] if siter < model.nsites - 1 else [])
            vindl = [model.upper_ind(sitel)] + ([model.bond(sitel - 1, sitel)] if sitel > 0 else [])
            
            lower_ind_prefix = model.lower_ind_id.rstrip("{}")
            lower_ind_l = [f"{lower_ind_prefix}{sitel}"] if f"{lower_ind_prefix}{sitel}" in list(model.lower_inds) else []
            lower_ind_r = [f"{lower_ind_prefix}{siter}"] if f"{lower_ind_prefix}{siter}" in list(model.lower_inds) else []
            if lower_ind_l:
                left_inds=[*vindl, *lower_ind_l]
            else:
                left_inds=[*vindl]

            if lower_ind_r:
                right_inds=[*vindr, *lower_ind_r]
            else:
                right_inds=[*vindr]
            splited_tensors = qtn.tensor_core.tensor_split(tensor, get='tensors', left_inds=left_inds, right_inds=right_inds, max_bond=model.max_bond(), **self.split_opts)
            # model.split_tensor(sitetags, left_inds=[*vindl, *lower_ind], max_bond=self.bond_dim_split, **self.split_opts)
        
        tids = model._get_tids_from_tags(sitetags, which='all')
        for tid in tuple(tids):
            model.pop_tensor(tid)
        #model.delete(sitetags)
        for t in splited_tensors:
            inds = t.inds
            if len(inds) == 4:
                t.transpose(inds[1], inds[3], inds[0], inds[2], inplace=True)
            elif len(inds) == 3 and (0 in sites or (model.L - 1) in sites):
                t.transpose(inds[2], inds[1], inds[0], inplace=True)
            elif len(inds) == 3:
                t.transpose(inds[2], inds[1], inds[0], inplace=True)
            # maybe add for len == 2
            model.add_tensor(t)

        # fix tags
        for tag in sitetags:
            for tensor in model.select_tensors(tag):
                tensor.drop_tags()
                site_ind = next(filter(lambda ind: ind.removeprefix(site_ind_prefix).isdecimal(), tensor.inds))
                site = site_ind.removeprefix(site_ind_prefix)
                tensor.add_tag(model.site_tag(site))     


class Global(Strategy):
    """Global optimization through Gradient Descent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def iterate_sites(self, model):
        yield model.sites

    def posthook(self, model, sites):
        # renormalize
        if self.renormalize:
            model.normalize(inplace=True)