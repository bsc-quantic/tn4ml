import itertools
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import TensorNetwork1DOperator, TensorNetwork1DFlat, TensorNetwork1D


class SpacedMatrixProductOperator(TensorNetwork1DOperator, TensorNetwork1DFlat, TensorNetwork1D, qtn.TensorNetwork):
    """A MatrixProductOperator with a decimated number of output indices.

    Parameters
    ----------
    spacing : int
        Spacing paramater, or space between output indices in number of sites.
    embed_dim : int
    physical_dim : int
    """

    _EXTRA_PROPS = ("_site_tag_id", "_upper_ind_id", "_lower_ind_id", "_L", "_spacing")

    def __init__(self, arrays, shape="lrud", site_tag_id="I{}", tags=None, upper_ind_id="k{}", lower_ind_id="b{}", bond_name="", **tn_opts):
        if isinstance(arrays, SpacedMatrixProductOperator):
            super().__init__(arrays)
            return

        arrays = tuple(arrays)
        self._L = len(arrays)

        # process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        upper_inds = map(upper_ind_id.format, range(self.L))
        lower_inds = map(lower_ind_id.format, range(self.L))

        # process tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(self.L))
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
            site_tags = tuple((st,) + tags for st in site_tags)

        self.cyclic = qu.ops.ndim(arrays[0]) == 4
        self.spacing = ([x.dim for x in arrays].count(4) + self.L - 1) // self.L

        # process orders
        lu_ord = tuple(shape.replace("r", "").replace("d", "").find(x) for x in "lu")
        lud_ord = tuple(shape.replace("r", "").find(x) for x in "lud")
        rud_ord = tuple(shape.replace("l", "").find(x) for x in "rud")
        lru_ord = tuple(shape.replace("u", "").find(x) for x in "lru")
        lrud_ord = tuple(map(shape.find, "lrud"))

        if self.cyclic:
            if (self.L - 1) % self.spacing == 0:
                last_ord = lrud_ord
            else:
                last_ord = lru_ord
        else:
            if (self.L - 1) % self.spacing == 0:
                last_ord = lud_ord
            else:
                last_ord = lu_ord

        orders = [rud_ord if not self.cyclic else lrud_ord, *[lrud_ord if i % self.spacing == 0 else lru_ord for i in range(1, self.L - 1)], last_ord]

        # process inds
        cyc_bond = (qtn.rand_uuid(base=bond_name),) if self.cyclic else ()
        nbond = qtn.rand_uuid(base=bond_name)

        inds = []
        inds += [(*cyc_bond, nbond, next(upper_inds), next(lower_inds))]
        pbond = nbond

        for i in range(1, self.L - 1):
            nbond = qtn.rand_uuid(base=bond_name)

            if i % self.spacing == 0:
                curr_down_id = [lower_ind_id.format(i)]
            else:
                curr_down_id = []

            ind += [(pbond, nbond, next(upper_inds), *curr_down_id)]
            pbond = nbond

        last_down_ind = [lower_ind_id.format(self.L - 1)] if (self.L - 1) % self.spacing == 0 else []
        ind += [(pbond, *cyc_bond, next(upper_inds), *last_down_ind)]

        tensors = [qtn.Tensor(data=qu.ops.transpose(array, order), inds=inds, tags=site_tag) for array, site_tag, inds, order in zip(arrays, site_tags, inds, orders)]

        super().__init__(tensors, virtual=True, **tn_opts)

    def rand(n: int, spacing: int, phys_dim: tuple[int, int] = 2, cyclic: bool = False, init_func="normal", **kwargs):
        arrays = []
        for i, hasoutput in zip(range(n), itertools.cycle([True, *[False] * (spacing - 1)])):
            if hasoutput:
                # TODO
                pass
            else:
                # TODO
                pass

        return SpacedMatrixProductOperator(arrays, **kwargs)

    @property
    def spacing(self) -> int:
        return self._spacing

    def apply():
        pass

    def trace():
        pass
