import itertools
import math
from typing import Tuple, Collection
import numpy as np

import autoray as a
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import TensorNetwork, TensorNetwork1DFlat, TensorNetwork1DOperator, MatrixProductState
from .model import Model
from ..util import return_digits

def sort_tensors(tn):
    """Helper function for sorting tensors of tensor network in alphabetic order by tags.

    Parameters
    ----------
    tn : :class:`quimb.tensor.tensor_core.TensorNetwork`
        Tensor network.

    Returns
    -------
    tuple
        Tuple of sorted tensors.
    """

    ts_and_sorted_tags = [(t, sorted(return_digits(t.tags))) for t in tn]
    ts_and_sorted_tags.sort(key=lambda x: x[1])
    return tuple(x[0] for x in ts_and_sorted_tags)

def gramschmidt(A):
    """Function that creates an orthogonal basis from a matrix `A`.

    Parameters
    ----------
    A : Matrix

    Returns
    -------
    `np.numpy.ndarray`
        Matrix in a orthogonal basis

    """
    m = A.shape[0]

    for i in range(m-1):
        v = [A[i, :]]
        v /= np.linalg.norm(v)
        A[i, :] = v

        sA = A[i+1:, :]
        u = np.matmul(sA, np.transpose(v))
        sA -= np.matmul(u, np.conjugate(v))
        A[i+1:, :] = sA
        u = np.matmul(sA, np.transpose(v))

    A[-1,:] /= np.linalg.norm(A[-1,:])
    return A

class SpacedMatrixProductOperator(TensorNetwork1DOperator, TensorNetwork1DFlat, Model):
    """A MatrixProductOperator with a decimated number of output indices.
    See :class:`quimb.tensor.tensor_1d.MatrixProductOperator` for explanation of other attributes and methods.
    """

    _EXTRA_PROPS = ("_site_tag_id", "_upper_ind_id", "_lower_ind_id", "_L", "_spacing", "_spacings", "_orders")

    def __init__(self, arrays, output_inds=[], shape="lrud", site_tag_id="I{}", tags=None, upper_ind_id="k{}", lower_ind_id="b{}", bond_name="bond{}", **tn_opts):
        if isinstance(arrays, SpacedMatrixProductOperator):
            TensorNetwork.__init__(self, arrays)
            return

        arrays = tuple(arrays)
        self._L = len(arrays)
        # process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        upper_inds = map(upper_ind_id.format, range(self.L))

        # process tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(self.L))
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
            site_tags = tuple((st,) + tags for st in site_tags)

        self.cyclic = qtn.array_ops.ndim(arrays[0]) == 4
        dims = [x.ndim for x in arrays]

        # if spacing is not even
        if output_inds:
            lower_inds = map(lower_ind_id.format, output_inds)
            self._spacing = 0
            self._spacings = [(o - output_inds[i]) for i, o in enumerate(output_inds[1:])]
            self._spacings.append(len(arrays) - 1 - output_inds[-1])
        else:
            # enable spacing == (to have one output)
            if dims.count(4) == 0:
                self._spacing = self.L
            elif dims.count(4) == 1:
                self._spacing = dims.index(4)
            else:
                self._spacing = dims.index(4, dims.index(4) + 1) - dims.index(4)
            self._spacings=[]
            lower_inds = map(lower_ind_id.format, range(0, self.L, self.spacing))

        # process orders
        lu_ord = tuple(shape.replace("r", "").replace("d", "").find(x) for x in "lu")
        lud_ord = tuple(shape.replace("r", "").find(x) for x in "lud")
        rud_ord = tuple(shape.replace("l", "").find(x) for x in "rud")
        lru_ord = tuple(shape.replace("u", "").find(x) for x in "lru")
        lrud_ord = tuple(map(shape.find, "lrud"))

        if self.cyclic:
            if output_inds:
                if (self.L - 1) in output_inds:
                    last_ord = lrud_ord
                else:
                    last_ord = lru_ord
            else:
                if (self.L - 1) % self.spacing == 0:
                    last_ord = lrud_ord
                else:
                    last_ord = lru_ord
        else:
            if output_inds:
                if (self.L - 1) in output_inds:
                    last_ord = lud_ord
                else:
                    last_ord = lu_ord
            else:
                if (self.L - 1) % self.spacing == 0:
                    last_ord = lud_ord
                else:
                    last_ord = lu_ord

        # orders = [rud_ord if not self.cyclic else lrud_ord, *[lrud_ord for i in range(1, self.L - 1)], last_ord]
        orders = [rud_ord if not self.cyclic else lrud_ord, *[lrud_ord if (output_inds and (i in output_inds)) or (self.spacing and i % self.spacing == 0) else lru_ord for i in range(1, self.L - 1)], last_ord]
        self._orders = orders

        # process inds
        bond_ids = list(range(0, self.L))
        # cyc_bond = (qtn.rand_uuid(base=bond_name),) if self.cyclic else ()
        cyc_bond = (f"bond_{self.L}",) if self.cyclic else ()
        # nbond = qtn.rand_uuid(base=bond_name)
        nbond = f"bond_{bond_ids[0]}"

        inds = []
        inds += [(*cyc_bond, nbond, next(upper_inds), next(lower_inds))]
        pbond = nbond

        for i in range(1, self.L - 1):
            # nbond = qtn.rand_uuid(base=bond_name)
            nbond = f"bond_{bond_ids[i]}"
            if output_inds:
                if i in output_inds:
                    curr_down_id = [lower_ind_id.format(i)]
                else:
                    curr_down_id = []
            else:
                if i % self.spacing == 0:
                    curr_down_id = [lower_ind_id.format(i)]
                else:
                    curr_down_id = []

            inds += [(pbond, nbond, next(upper_inds), *curr_down_id)]
            pbond = nbond

        last_down_ind = [lower_ind_id.format(self.L - 1)] if (output_inds and ((self.L-1) in output_inds)) or (self.spacing and ((self.L - 1) % self.spacing == 0)) else []
        inds += [(pbond, *cyc_bond, next(upper_inds), *last_down_ind)]
        tensors = [qtn.Tensor(data=a.transpose(array, order), inds=ind, tags=site_tag) for array, site_tag, ind, order in zip(arrays, site_tags, inds, orders)]
        TensorNetwork.__init__(self, tensors, virtual=True, **tn_opts)

    def normalize(self, insert=None):
        """Function for normalizing tensors of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        insert : int
            Index of tensor divided by norm. *Default = None*. When `None` the norm division is distributed across all tensors.
        """
        norm = self.norm()
        if insert == None:
            for tensor in self.tensors:
                tensor.modify(data=tensor.data / a.do("power", norm, 1 / self.L))
        else:
            self.tensors[insert].modify(data=self.tensors[insert].data / norm)

    def norm(self, **contract_opts):
        """Calculates norm of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        contract_opts : Optional
            Arguments passed to ``contract()``.

        Returns
        -------
        float
            Norm of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        """
        norm = self.conj() & self
        return norm.contract(**contract_opts) ** 0.5

    def rand_distribution(n: int, spacing: int, bond_dim: int = 4, phys_dim: Tuple[int, int] = (2, 2), cyclic: bool = False, init_func: str = "uniform", scale: float = 1.0, seed: int = None, insert = 0, **kwargs):
        """Generates :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` with random tensor arrays.

        Parameters
        ----------
        n: int
            Number of tensors.
        spacing : int
            Spacing paramater, or space between output indices in number of sites.
        bond_dim : int
            Dimension of virtual indices between tensors. *Default = 4*.
        phys_dim :  tuple(int, int)
            Dimension of physical indices for individual tensor - *up* and *down*.
        cyclic : bool
            Flag for indicating if SpacedMatrixProductOperator is cyclic. *Default=False*.
        init_func : str
            Type of random number for generating arrays data. *Default='uniform'*.
        scale : float
            The width of the distribution (standard deviation if `init_func='normal'`).
        seed : int, or `None`
            Seed for generating random number.
        insert : int
            Index of tensor divided by norm. *Default = 0*. When `None` the norm division is distributed across all tensors.

        Returns
        -------
        :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        """

        arrays = []
        for i, hasoutput in zip(range(n), itertools.cycle([True, *[False] * (spacing - 1)])):
            if hasoutput:
                shape = (bond_dim, bond_dim, *phys_dim)
                if not cyclic:
                    if i == 0:
                        shape = (bond_dim, *phys_dim)
                    if i == n - 1:
                        shape = (bond_dim, *phys_dim)
            else:
                shape = (bond_dim, bond_dim, phys_dim[0])
                if i == n - 1 and not cyclic:
                    shape = (bond_dim, phys_dim[0])
            if seed != None:
                arrays.append(qu.gen.rand.randn(shape, dist=init_func, scale=scale, seed=seed))
            else:
                arrays.append(qu.gen.rand.randn(shape, dist=init_func, scale=scale))
        mpo = SpacedMatrixProductOperator(arrays, **kwargs)
        mpo.compress(form="flat", max_bond=bond_dim)  # limit bond_dim

        for i, tensor in enumerate(mpo.tensors):
            tensor_norm = tensor.norm()
            tensor.modify(data=tensor.data / np.sqrt(tensor_norm))

        if insert == None:
            mpo.normalize(insert)
        else:
            mpo.canonize(insert)
            mpo.normalize(insert)

        return mpo

    def rand_orthogonal(n: int, spacing: int = None, bond_dim: int = 4, phys_dim: Tuple[int, int] = (2, 2), output_inds: Collection = [], cyclic: bool = False, init_func: str = "uniform", scale: float = 1.0, seed: int = None, **kwargs):
        """Generates :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` with random tensors in a
        orthogonal basis, which fulfill that the `tn4ml.models.smpo.SpacedMatrixProductOperator` is
        normalized. Currently this function is only supported for `cyclic=False`.

        Parameters
        ----------
        n: int
            Number of tensors.
        spacing : int
            Spacing paramater, or space between output indices in number of sites.
        bond_dim : int
            Dimension of virtual indices between tensors. *Default = 4*.
        phys_dim :  tuple(int, int)
            Dimension of physical indices for individual tensor - *up* and *down*.
        output_inds : array of int
            Indexes of tensors which have output indices. From 0 to n.
        cyclic : bool
            Flag for indicating if SpacedMatrixProductOperator is cyclic. *Default=False*.
        init_func : str
            Type of random number for generating arrays data. *Default='uniform'*.
        scale : float
            The width of the distribution (standard deviation if `init_func='normal'`).
        seed : int, or `None`
            Seed for generating random number.

        Returns
        -------
        :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        """

        if cyclic:
            raise NotImplementedError()

        if 0 not in output_inds:
            raise ValueError("First tensor needs to have output index.")

        if output_inds:
            hasoutput = []
            for i in range(n):
                if i in output_inds: hasoutput.append(True)
                else: hasoutput.append(False)
            spacings = [(o - output_inds[i]) for i, o in enumerate(output_inds[1:])]
            spacings.append(n - 1 - output_inds[-1])
        else:
            hasoutput = itertools.cycle([True, *[False] * (spacing - 1)])

        arrays = []; h = 0
        for i, has_out in zip(range(1, n+1), hasoutput):
            if i > n // 2:
                j = (n + 1 - abs(2*i - n - 1)) // 2
            else:
                j = i
            
            if output_inds:
                if len(spacings)-1 >= h:
                    spacing = spacings[h]
                if has_out:
                    h+=1

            chil = min(bond_dim, phys_dim[0] ** (j-1) * phys_dim[1] ** ((j-1)//spacing))
            chir = min(bond_dim, phys_dim[0] ** (j) * phys_dim[1] ** ((j)//spacing))

            if i > n // 2:
                (chil, chir) = (chir, chil)

            if has_out:
                if i == 1:
                    shape = (chir, *phys_dim)
                elif i == n:
                    shape = (chil, *phys_dim)
                else:
                    shape = (chil, chir, *phys_dim)
            else:
                if i == 1:
                    shape = (chir, phys_dim[0])
                elif i == n:
                    shape = (chil, phys_dim[0])
                else:
                    shape = (chil, chir, phys_dim[0])

            if seed != None:
                A = gramschmidt(qu.gen.rand.randn([shape[0], np.prod(shape[1:])], dist=init_func, scale=scale, seed=seed))
            else:
                A = gramschmidt(qu.gen.rand.randn([shape[0], np.prod(shape[1:])], dist=init_func, scale=scale))
            arrays.append(np.reshape(A, shape))

        arrays[0] /= np.sqrt(min(bond_dim, phys_dim[0]))

        mpo = SpacedMatrixProductOperator(arrays, output_inds, **kwargs)
        return mpo

    @property
    def spacing(self) -> int:
        """Spacing paramater, or space between output indices in number of sites.
        """
        return self._spacing
    
    @property
    def spacings(self) -> list:
        """Spacings paramater, or space between output indices in number of sites.
        """
        return self._spacings

    @property
    def lower_inds(self):
        return map(self.lower_ind, range(0, self.L, self.spacing))

    def get_orders(self) -> list:
        return self._orders

    def apply_mps(tn_op, tn_vec, compress=False, **compress_opts):
        """Version of :func:`quimb.tensor.tensor_1d.MatrixProductOperator._apply_mps()` for :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        tn_op : :class:`quimb.tensor.tensor_core.TensorNetwork`
            The tensor network representing the operator.
        tn_vec : :class:`quimb.tensor.tensor_core.TensorNetwork`, or :class:`quimb.tensor.tensor_1d.MatrixProductState`
            The tensor network representing the vector.
        compress : bool
            Whether to compress the resulting tensor network.
        compress_opts: optional
            Options to pass to ``tn_vec.compress``.

        Returns
        -------
        :class:`quimb.tensor.tensor_1d.MatrixProductState`
        """

        smpo, mps = tn_op.copy(), tn_vec.copy()
        if smpo.spacings:
            spacings = smpo.spacings
        else:
            spacings = [smpo.spacing]*len(list(smpo.lower_inds))
            if list(smpo.lower_inds)[-1] != len(smpo.tensors)-1:
                spacings.append(len(smpo.tensors)-1-list(smpo.lower_inds)[-1])

        # align the indices
        coordinate_formatter = qu.tensor.tensor_arbgeom.get_coordinate_formatter(smpo._NDIMS)
        smpo.lower_ind_id = f"__tmp{coordinate_formatter}__"
        smpo.upper_ind_id = mps.site_ind_id

        result = smpo & mps

        for ind in mps.outer_inds():
            result.contract_ind(ind=ind)

        list_tensors = result.tensors
        number_of_sites = len(list_tensors)
        tags = list(qtn.tensor_core.get_tags(result))

        i = 0
        for s_i, S in enumerate(spacings):
            if S > 1:
                tags_to_drop = []
                for j in range(i + 1, i + S):
                    if j + 1 == i + S or j >= number_of_sites - 1:
                        break
                    result.contract_ind(list_tensors[j].bonds(list_tensors[j + 1]))
                    tags_to_drop.extend([tags[j], tags[j + 1]])
                if i + 1 == len(tags) and list_tensors[i].ndim != 2:
                    # if last site of smpo has output_ind
                    break
                result.contract_ind(list_tensors[i].bonds(list_tensors[i + 1]))
                if len(tags_to_drop) == 0:
                    tags_to_drop.append(tags[i + 1])
                result.drop_tags(tags_to_drop)

                i = i + S
            result.fuse_multibonds_()

        sorted_tensors = sort_tensors(result)
        arrays = [tensor.data for tensor in sorted_tensors]

        if len(arrays[0].shape) == 3:
            arr = np.squeeze(arrays[0])
            arrays[0] = arr
        arrays[0] = a.do("reshape", arrays[0], (1, *arrays[0].shape))

        if len(arrays[-1].shape) == 3:
            arr = np.squeeze(arrays[-1])
            arrays[-1] = arr

        if S == 1:
            arrays[-1] = a.do("reshape", arrays[-1], (arrays[-1].shape[0], 1, arrays[-1].shape[1]))
        else:
            arrays[-1] = a.do("reshape", arrays[-1], (*arrays[-1].shape, 1))

        # set shape
        if len(spacings) == 1 and spacings[0]==1: shape = 'lrp'
        else: shape = 'lpr'

        vec = MatrixProductState(arrays, shape=shape)
        # optionally compress
        if compress:
            vec.compress(**compress_opts)
        return vec

    def apply_smpo(tn_op_1, tn_op_2, compress=False, **compress_opts):
        """Version of :func:`quimb.tensor.tensor_1d.MatrixProductOperator._apply_mpo()` for :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` - computes trace.

        Parameters
        ----------
        tn_op_1: :class:`quimb.tensor.tensor_core.TensorNetwork`
            The tensor network representing the operator 1.
        tn_op_2: :class:`quimb.tensor.tensor_core.TensorNetwork`
            The tensor network representing the operator 2.
        compress: bool
            Whether to compress the resulting tensor network.
        compress_opts: optional
            Options to pass to ``tn_vec.compress``.

        Returns
        -------
        :class:`quimb.tensor.tensor_1d.MatrixProductOperator`
        """

        # assume that A and B have same spacing
        assert tn_op_1.spacing == tn_op_2.spacing

        A, B = tn_op_1.copy(), tn_op_2.copy()

        tn = A | B

        for tag in A.site_tags:
            tn.contract_tags([tag], inplace=True)

        # optionally compress
        if compress:
            tn.compress(**compress_opts)

        trace = tn.contract_cumulative(tn.site_tags)
        return trace

    def apply(self, other, compress=False, **compress_opts):
        """Version of :func:`quimb.tensor.tensor_1d.MatrixProductOperator.apply`for :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.
        Act with this SMPO on another SMPO or MPS, such that the resulting
        object has the same tensor network structure/indices as ``other``.
        For an MPS:

                   |  S  |  S  |  S  |  S  |  S  |
             self: A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A  where S = spacing
                   | | | | | | | | | | | | | | | |
            other: x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x

                                   -->

                   |  S  |  S  |  S  |  S  |  S  |   <- other.site_ind_id
              out: y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y
        For an SMPO:

                   | | | | | | | | | | | | | | | | <- self.upper_ind_id
             self: A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A
                   |  S  |  S  |  S  |  S  |  S  | <- lower_ind_id
            other: B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B
                   | | | | | | | | | | | | | | | | <- other.upper_ind_id

                                   -->

                   | | | | | | | | | | | | | | | | | |   <- other.upper_ind_id
              out: C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C
                   | | | | | | | | | | | | | | | | | |   <- other.lower_ind_id

        The resulting TN will have the same structure/indices as ``other``, but
        probably with larger bonds (depending on compression).


        Parameters
        ----------
        other : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`, or :class:`quimb.tensor.tensor_1d.MatrixProductState`.
            The object to act on.
        compress : bool
            Whether to compress the resulting object.
        compress_opts: optional
            Supplied to :meth:`TensorNetwork1DFlat.compress`.

        Returns
        -------
        :class:`quimb.tensor.tensor_1d.MatrixProductOperator`, or :class:`quimb.tensor.tensor_1d.MatrixProductState`
        """

        if isinstance(other, MatrixProductState):
            return self.apply_mps(other, compress=compress, **compress_opts)
        elif isinstance(other, SpacedMatrixProductOperator):
            return self.apply_smpo(other, compress=compress, **compress_opts)
        else:
            raise TypeError("Can only Dot with a SpacedMatrixProductOperator or a " f"MatrixProductState, got {type(other)}")
