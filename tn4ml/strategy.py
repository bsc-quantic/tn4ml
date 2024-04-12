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
    """DMRG-like local optimization."""

    def __init__(self, grouping: int = 2, two_way=True, split_opts={"cutoff": 0.}, **kwargs):

        """Constructor for Sweeps strategy.
        
        Attributes
        ----------
        grouping : int
            Number of tensors to group together. *Default*=**2**.
        two_way : bool
            Flag indicating wheather sweeping happens two-way or one-way. *Default*=**True** *(two-way sweep)*.
        split_opts: optional
            Additional args passed to ``model.split_tensor()``.
        kwargs : optional
            Additional keyword arguments passed to inherited class.
        
        Raises
        ------
        ValueError
            If `grouping` > 2.
        ValueError
            If `grouping` == 1.
        """
        # TODO right now only support grouping <= 2
        if grouping > 2:
            raise ValueError(f"grouping - {grouping=} > 2")
        if grouping == 1:
            raise ValueError("grouping == 1")
        
        self.grouping = grouping
        self.two_way = two_way
        self.split_opts = split_opts
        self.inds_order = dict() # remember order of inds - on first sweep
        super().__init__(**kwargs)

    def iterate_sites(self, model):
        _check_model(model)
        sites = []
        # forward
        for i in model.sites[: len(model.sites) - self.grouping + 1]:
            sites.append(tuple(model.sites[i + j] for j in range(self.grouping)))

        # backward
        if self.two_way:
            for i in list(reversed(model.sites[self.grouping - 1:])):
                sites.append(tuple(model.sites[i - j] for j in range(self.grouping)))

        for site in sites:
            yield site

    def prehook(self, model, sites):
        """Contract tensors before computing gradients.
        
        Parameters
        ----------
        model : :class:`tn4ml.models.Model``
            Model
        sites : sequence of int
            List of tensor ids.
        
        Raises
        ------
        NotImplementedError
            If `grouping` > 2.
        """
        _check_model(model)

        if self.grouping > 2:
            raise NotImplementedError('Not implememented for grouping > 2.')
        
        model.canonize(sites)
        
        # remembed bond_size and bond_name
        self.bond_dim_split = model.bond_size(sites[0], sites[1])
        self.bond_name = model.bond(sites[0], sites[1])

        sitetags = tuple(model.site_tag(site) for site in sites)

        if self.two_way:
            self.left_inds = model.select_tensors(sitetags[0])[0].inds
            self.right_inds = model.select_tensors(sitetags[1])[0].inds
        else:
            self.left_inds = model.select_tensors(sitetags[1])[0].inds
            self.right_inds = model.select_tensors(sitetags[0])[0].inds
        
        model.contract_tags(sitetags, output_inds = self.inds_order[sites] if sites in self.inds_order.keys() else None, inplace=True)

        # remember order of inds
        if sites not in self.inds_order.keys():
            sitetags = [model.site_tag(site) for site in sites]
            self.inds_order[sites] = model.select_tensors(sitetags)[0].inds

    def posthook(self, model, sites):
        """Split tensors after computing gradients.
        
        Parameters
        ----------
        model : :class:`tn4ml.models.Model``
            Model
        sites : sequence of `str`
            List of tensors' tags.
        
        Raises
        ------
        ValueError
            If split function didn't produce correct tensors.
        """
        _check_model(model)
        # get tensor
        sitetags = [model.site_tag(site) for site in sites]
        tensor = model.select_tensors(sitetags, which="all")[0]

        # normalize
        if self.renormalize:
            tensor.normalize(inplace=True)

        # split tensor with DMRG
        sitel, siter = sites
        if  self.two_way and sitel > siter:
            siter, sitel = sites
        
        if isinstance(model, qtn.MatrixProductState): # TODO - fix! not working
            site_ind_prefix = model.site_ind_id.rstrip("{}")
            vindl = [f'{site_ind_prefix}{sitel}'] + ([model.bond(sitel - 1, sitel)] if sitel > 0 else [])
            vindr = [f'{site_ind_prefix}{siter}']
            qtn.tensor_core.tensor_split(tensor, left_inds=vindl, right_inds=vindr, bond_ind=self.bond_name, max_bond=self.bond_dim_split, **self.split_opts)
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
            
            splited_tensors = qtn.tensor_core.tensor_split(tensor, get='tensors', left_inds=left_inds, right_inds=right_inds, bond_ind = self.bond_name, max_bond=self.bond_dim_split, **self.split_opts)

            tids = model._get_tids_from_tags(sitetags, which='all')
            for tid in tuple(tids):
                model.pop_tensor(tid)
                
            # transpose to LRP order
            for t in splited_tensors:
                inds = t.inds
                inds_len = len(inds)
                if inds_len in [2, 3, 4]:
                    for direction in [self.left_inds, self.right_inds]:
                        if inds_len == len(direction) and sorted(inds) == sorted(direction):
                            t.transpose(*direction[:inds_len], inplace=True)
                            break
                else:
                    raise ValueError('Something is wrong in index ordering!')

                model.add_tensor(t)
        # fix tags
        for tag in sitetags:
            for tensor in model.select_tensors(tag):
                tensor.drop_tags()
                site_ind = next(filter(lambda ind: ind.removeprefix(site_ind_prefix).isdecimal(), tensor.inds))
                site = site_ind.removeprefix(site_ind_prefix)
                tensor.add_tag(model.site_tag(site))


# not used
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

def _check_model(model):
    if not all(hasattr(model, attr) for attr in ['sites', 'canonize', 'bond_size', 'bond', 'site_tag', 'select_tensors']):
        raise TypeError("model object doesn't have necessary methods or properties")