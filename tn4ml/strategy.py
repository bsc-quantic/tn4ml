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
        """Function for iterating selected tensors.

        Parameters
        ----------
        sites : sequence of `str`
            List of tensors' tags.
        """
        pass


class Sweeps(Strategy):
    """
    The sweeping DMRG (Density Matrix Renormalization Group) technique is an algorithm used to efficiently find the ground state of large quantum systems.
    But in general in Machine Learning, it is used to optimize the parameters of a tensor network model.
    It works by iteratively optimizing the parameters, focusing on local regions and gradually improving the accuracy of the solution.

    Sweeping Process:

    - Left-to-Right Sweep:
       Contract two tensors into one, find the gradient of the loss function with respect to that contracted tensor,
       update the parameter of concatenated tensor, and then split the tensor back into two.
       Swipe from first to last tensor in the tensor network.
    - Right-to-Left Sweep:
        Same process as left-to-right sweep but in the opposite direction.

    Iterative Refinement:
        Repeat the left-to-right and right-to-left sweeps multiple times.
        Each iteration (or sweep) improves the overall accuracy of the optimization.

    Convergence:
        The process continues until the changes in the parameters become negligible.
    """

    def __init__(
        self, grouping: int = 2, two_way=True, split_opts={"cutoff": 0.0}, **kwargs
    ):
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

        if self.grouping == 2:
            self.split_opts = split_opts
            self.inds_order = dict()  # remember order of inds
            self.bond_dim_split = None  # remember bond size
            self.bond_name = None  # remember bond name

        super().__init__(**kwargs)

    def iterate_sites(self, model):
        _check_model(model)
        sites = []
        # forward
        for i in model.sites[: len(model.sites) - self.grouping + 1]:
            sites.append(tuple(model.sites[i + j] for j in range(self.grouping)))

        # backward
        if self.two_way:
            for i in list(reversed(model.sites[self.grouping - 1 :])):
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
            raise NotImplementedError("Not implememented for grouping > 2.")

        model.canonicalize(set(sites), inplace=True)

        if self.grouping == 2:
            self.bond_dim_split = model.bond_size(sites[0], sites[1])
            self.bond_name = model.bond(sites[0], sites[1])

            sitetags = tuple(model.site_tag(site) for site in sites)

            if self.two_way:
                self.left_inds = model.select_tensors(sitetags[0])[0].inds
                self.right_inds = model.select_tensors(sitetags[1])[0].inds
            else:
                self.left_inds = model.select_tensors(sitetags[1])[0].inds
                self.right_inds = model.select_tensors(sitetags[0])[0].inds

            # Always contract without enforcing output_inds: canonicalize
            # auto-renames bond indices each pass, so stored names go stale.
            model.contract_tags(sitetags, inplace=True)

            # Always refresh inds_order after contraction so posthook and
            # model.py see the current bond names, not stale ones.
            key = sites
            sitetags = [model.site_tag(site) for site in sites]
            self.inds_order[key] = model.select_tensors(sitetags)[0].inds

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

        if self.grouping == 2:
            # split tensor into two
            sitel, siter = sites
            if self.two_way and sitel > siter:
                siter, sitel = sites

            bond_ind = "bond_{}".format(sitel)

            # Use the index sets saved in prehook (captured after canonicalize).
            # _get_inds_for_split looks up bond names by convention (bond_N) but
            # canonicalize may rename them to auto-generated ids, causing the split
            # to miss the external bond index.
            #
            # self.left_inds/right_inds map to sites[0]/sites[1] for two_way forward,
            # but are swapped for backward sweeps and one_way (see prehook).
            if self.two_way and sites[0] < sites[1]:
                left_inds = [ind for ind in self.left_inds if ind != self.bond_name]
                right_inds = [ind for ind in self.right_inds if ind != self.bond_name]
            else:
                left_inds = [ind for ind in self.right_inds if ind != self.bond_name]
                right_inds = [ind for ind in self.left_inds if ind != self.bond_name]

            splited_tensors = qtn.tensor_core.tensor_split(
                tensor,
                get="tensors",
                left_inds=left_inds,
                right_inds=right_inds,
                bond_ind=bond_ind,
                max_bond=self.bond_dim_split,
                **self.split_opts,
            )

            # remove old tensor from the network
            tids = model._get_tids_from_tags(sitetags, which="all")
            for tid in tuple(tids):
                model.pop_tensor(tid)

            expected_inds = self.inds_order[sites]

            # match both tensors using index sets
            for i, t in enumerate(splited_tensors):
                if sorted(t.inds) == sorted(expected_inds):
                    splited_tensors[i].transpose(*expected_inds, inplace=True)
                else:
                    other_inds = list(set(t.inds) - set(expected_inds))
                    new_order = [
                        ix for ix in t.inds if ix not in other_inds
                    ] + other_inds
                    splited_tensors[i].transpose(*new_order, inplace=True)

            # fix tags BEFORE adding back
            for site, tensor in zip(sorted(sites), splited_tensors):
                tensor.drop_tags()
                tensor.add_tag(model.site_tag(site))
                model.add_tensor(tensor)


# not used
class Global(Strategy):
    """Global optimization through Gradient Descent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def iterate_sites(self, model):
        yield model.sites

    def posthook(self, model, sites):
        # renormalize
        if self.renormalize:
            model.normalize(inplace=True)


def _check_model(model):
    if not all(
        hasattr(model, attr)
        for attr in [
            "sites",
            "canonize",
            "bond_size",
            "bond",
            "site_tag",
            "select_tensors",
        ]
    ):
        raise TypeError("Model object doesn't have necessary methods or properties")


def _get_inds_for_split(
    ind_map,
    sitel,
    siter,
    nsites,
    upper_ind_id="k{}",
    bond_ind_id="bond_{}",
    lower_ind_id="b{}",
):

    # normalize order
    if sitel > siter:
        sitel, siter = siter, sitel

    def _idx_exists(ind_name):
        return ind_name in ind_map

    # upper index (input/output) per site
    ul = upper_ind_id.format(sitel)
    ur = upper_ind_id.format(siter)

    # lower (optional)
    ll = lower_ind_id.format(sitel)
    lr = lower_ind_id.format(siter)

    # bonds: left of sitel, and right of siter
    # i.e., bond before sitel and after siter if they exist
    bl = (
        bond_ind_id.format(sitel - 1)
        if bond_ind_id.format(sitel - 1) in ind_map
        else None
    )
    br = (
        bond_ind_id.format(siter)
        if bond_ind_id.format(siter) in ind_map and siter < nsites
        else None
    )

    # bond connecting the two sites
    mid_bond = bond_ind_id.format(sitel)

    # build lists of indices
    vindl = [ul]
    if _idx_exists(bl):
        vindl.append(bl)
    if _idx_exists(ll):
        vindl.append(ll)

    vindr = [ur]
    if _idx_exists(br):
        vindr.append(br)
    if _idx_exists(lr):
        vindr.append(lr)

    return vindl, vindr, mid_bond
