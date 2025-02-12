import itertools
from typing import Tuple, Collection, Any
import numpy as np
import autoray as a

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, TensorNetwork1DOperator, TensorNetwork1DFlat

from jax.nn.initializers import Initializer
import jax.numpy as jnp

from .model import Model
from ..initializers import *
from ..util import return_digits

def sort_tensors(tn: qtn.TensorNetwork) -> tuple:
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

class SpacedMatrixProductOperator(TensorNetwork1DOperator, TensorNetwork1DFlat, Model):
    """A MatrixProductOperator with a decimated number of output indices.
    See :class:`quimb.tensor.tensor_1d.MatrixProductOperator` for explanation of other attributes and methods.
    """

    _EXTRA_PROPS = ("_site_tag_id", "_upper_ind_id", "_lower_ind_id", "_L", "_spacing", "_orders", "_spacings", "cyclic")

    def __init__(self, arrays, output_inds=[], shape="lrud", site_tag_id="I{}", tags=None, upper_ind_id="k{}", lower_ind_id="b{}", bond_name="bond{}", **tn_opts) -> None:
        """
        Create a MatrixProductOperator with a decimated number of output indices.
        
        Attributes
        ----------
        arrays : tuple of array_like
            The arrays defining the operator.
        output_inds : array of int
            Indexes of tensors which have output indices. From 0 to n. If spacing is not even.
        shape : str
            The shape of the tensors, e.g. 'lurd' or 'lrud'.
        site_tag_id : str
            The format string for the site tags.
        tags : str or sequence of str
            Global tags to add to all tensors.
        upper_ind_id : str
            The format string for the upper virtual indices.
        lower_ind_id : str
            The format string for the lower virtual indices.
        bond_name : str
            The format string for the bond names.
        tn_opts : optional
            Supplied to :class:`quimb.tensor.tensor_core.TensorNetwork`.
        """
        
        Model.__init__(self)

        if isinstance(arrays, SpacedMatrixProductOperator):
            qtn.TensorNetwork.__init__(self, arrays)
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
            if (len(arrays) - 1 - output_inds[-1]) != 0:
                self._spacings.append(len(arrays) - 1 - output_inds[-1])
        else:
            # enable spacing == (to have one output)
            if dims.count(4) == 0:
                if dims[-1] != 3:
                    self._spacing = self.L
                else:
                    self._spacing = self.L - 1
            elif dims.count(4) == 1:
                if dims.index(4) == 0:
                    self._spacing = self.L
                else:
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
        qtn.TensorNetwork.__init__(self, tensors, virtual=True, **tn_opts)

    def normalize(self, insert=None, output_inds=None) -> None:
        """Function for normalizing tensors of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        insert : int
            Index of tensor divided by norm. *Default = None*. When `None` the norm division is distributed across all tensors.
        """

        if self.L > 200:  # for large systems
            for i, tensor in enumerate(self.tensors):
                if i == 0:
                    self.left_canonize_site(i)
                elif i == self.L - 1:
                    tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
                else:
                    tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
                    self.left_canonize_site(i)
            jax.debug.print("{x}", x=self.norm())
        else:
            norm = self.norm(output_inds=output_inds)

            if insert == None:
                for tensor in self.tensors:
                    tensor.modify(data=tensor.data / a.do("power", norm, 1 / self.L))
            else:
                self.tensors[insert].modify(data=self.tensors[insert].data / norm)

    def norm(self, **contract_opts) -> float:
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

    # def copy(self):
    #     """Copies the model.
        
    #     Returns
    #     -------
    #     Model of the same type.
    #     """

    #     model = type(self)(self.arrays)
    #     for key in self.__dict__.keys():
    #         model.__dict__[key] = self.__dict__[key]
    #     return model

    def apply_mps(tn_op, tn_vec, normalize_on_contract=True, compress=False, **compress_opts):
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
            spacings = [smpo.spacing]*(len(list(smpo.lower_inds)))

        # align the indices
        coordinate_formatter = qu.tensor.tensor_arbgeom.get_coordinate_formatter(smpo._NDIMS)
        smpo.lower_ind_id = f"__tmp{coordinate_formatter}__"
        smpo.upper_ind_id = mps.site_ind_id

        result = smpo | mps

        for ind in mps.outer_inds():
            result.contract_ind(ind=ind)
        
        list_tensors = result.tensors
        number_of_sites = len(list_tensors)
        tags = list(qtn.tensor_core.get_tags(result))

        i = 0
        for S in spacings:
            if S > 1:
                tags_to_drop = []
                for j in range(i + 1, i + S):
                    if j >= number_of_sites - 1:
                        break
                    if len(list(list_tensors[j].tags)) > 1:
                        result.contract_ind(list_tensors[j].bonds(list_tensors[j + 1]))
                        for tag in list(list_tensors[j].tags):
                            tags_to_drop.extend([tag])
                    else:
                        result.contract_between(tags[j], tags[j + 1])
                        tags_to_drop.extend([tags[j]])
                    if normalize_on_contract:
                        result.normalize()
                if i + 1 == len(tags):
                    # if last site of smpo has output_ind
                    break
                result.drop_tags(tags_to_drop)
                i = i + S
            
            result.fuse_multibonds_()
        
        # if last tensor is a vector, contract it to previous one
        for t in result.tensors:
            if len(t.shape) == 1:
                result.contract_ind(list(t.inds))
                
                if normalize_on_contract:
                    result.normalize()

        sorted_tensors = sort_tensors(result)
        
        arrays = [tensor.data for tensor in sorted_tensors]

        if len(arrays[0].shape) == 3:
            if arrays[0].shape[0] != 1:
                arr = np.squeeze(arrays[0])
                if len(arr.shape) == 2:
                    arrays[0] = arr
                elif len(arr.shape) == 1: # weird
                    arrays[0] = a.do("reshape", arr, (*arr.shape, 1))
            else:
                arr = np.squeeze(arrays[0])
                arrays[0] = arr
            
        if len(arrays[-1].shape) == 3:
            arr = np.squeeze(arrays[-1])
            if len(arr.shape) == 1:
                arrays[-1] = a.do("reshape", arr, (*arr.shape, 1))
            else:
                arrays[-1] = arr
        elif len(arrays[-1].shape) == 1:
            arrays[-1] = a.do("reshape", arrays[-1], (*arrays[-1].shape, 1))

        for i, arr in enumerate(arrays):
            if len(arr.shape) >= 4:
                arr = np.squeeze(arr)
                if len(arr.shape) == 2 and i not in [0, len(arrays)-1]:
                    arrays[i] = a.do("reshape", arr, (*arr.shape, 1))
                else:
                    arrays[i] = arr

        shape = 'lrp'
        vec = MatrixProductState(arrays, shape=shape)
        
        # optionally compress
        if compress:
            vec.compress(**compress_opts)

        return vec

    def apply_smpo(tn_op_1, tn_op_2, trace=True, compress=False, **compress_opts):
        """Version of :func:`quimb.tensor.tensor_1d.MatrixProductOperator._apply_mpo()` for :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        tn_op_1: :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
            The tensor network representing the operator 1.
        tn_op_2: :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
            The tensor network representing the operator 2.
        compress: bool
            Whether to compress the resulting tensor network.
        compress_opts: optional
            Options to pass to ``tn_vec.compress``.

        Returns
        -------
        :class:`quimb.tensor.tensor_1d.TensorNetwork1D`
        """

        # assume that A and B have same spacing
        assert tn_op_1.spacing == tn_op_2.spacing # if self.spacings then self.spacing = 0

        A, B = tn_op_1.copy(), tn_op_2.copy()

        tn = A | B

        for tag in A.site_tags:
            tn.contract_tags([tag], inplace=True)
            

        # optionally compress
        if compress:
            tn.compress(**compress_opts)

        return tn

    def apply(self, other, normalize_on_contract=False, compress=False, **compress_opts):
        """
        Version of :func:`quimb.tensor.tensor_1d.MatrixProductOperator.apply` for :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.
        Act with this SMPO on another SMPO or MPS, such that the resulting
        object has the same tensor network structure/indices as `other`.
        For an MPS:

            .. image:: ../_static/smpo_mps.png
                    :width: 500px
                    :height: 250px
                    :scale: 80 %
                    :alt: Contraction of SMPO with MPS
                    :align: center
        
        For an SMPO:
        
            .. image:: ../_static/smpo_smpo.png
                    :width: 500px
                    :height: 250px
                    :scale: 80 %
                    :alt: Contraction of SMPO with SMPO
                    :align: center

        The resulting TN will have the same structure/indices as `other`, but probably with larger bonds (depending on compression).

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
            return self.apply_mps(other, normalize_on_contract=normalize_on_contract, compress=compress, **compress_opts)
        elif isinstance(other, SpacedMatrixProductOperator):
            return self.apply_smpo(other, compress=compress, **compress_opts)
        else:
            raise TypeError("Can only Dot with a SpacedMatrixProductOperator, MatrixProductOperator or a " f"MatrixProductState, got {type(other)}")
        
def generate_shape(method: str,
                L: int,
                has_out: bool = False,
                bond_dim: int = 2,
                phys_dim: Tuple[int, int] = (2, 2),
                cyclic: bool = False,
                position: int = None,
                spacing: int = None,
                ) -> tuple:
    """Returns a shape of tensor .

    Parameters
    ----------
    method : str
        Method on how to create shapes of tensors.
    L : int
        Number of tensors.
    has_out : bool
        Flag indicating does this tensor have an output index.
    bond_dim : int
        Dimension of virtual indices between tensors. *Default = 4*.
    phys_dim :  tuple(int, int)
        Dimension of physical indices for individual tensor - *up* and *down*.
    cyclic : bool
        Flag for indicating if SpacedMatrixProductOperator this tensor is part of is cyclic. *Default=False*.
    position : int
        Position of tensor in SpacedMatrixProductOperator.
    spacing : int
        Spacing paramater, or space between output indices in SpacedMatrixProductOperator. When spacing is even.
    Returns
    -------
        tuple
    """
        
    if method == 'even':
        # supported both for cyclic and non-cyclic
            if has_out:
                shape = (bond_dim, bond_dim, *phys_dim)
                if not cyclic:
                    if position == 1:
                        shape = (1, bond_dim, *phys_dim)
                    if position == L:
                        shape = (bond_dim, 1, *phys_dim)
            else:
                shape = (bond_dim, bond_dim, phys_dim[0])
                if position == 1 and not cyclic:
                    shape = (1, bond_dim, phys_dim[0])
                if position == L and not cyclic:
                    shape = (bond_dim, 1, phys_dim[0])
    else:
        assert not cyclic
        if position > L // 2:
            j = (L + 1 - abs(2*position - L - 1)) // 2
        else:
            j = position

        chir = min(bond_dim, phys_dim[0] ** (j) * phys_dim[1] ** ((j)//spacing))
        chil = min(bond_dim, phys_dim[0] ** (j-1) * phys_dim[1] ** ((j-1)//spacing))

        if position > L // 2:
            (chil, chir) = (chir, chil)

        if has_out:
            if position == 1:
                shape = (chir, *phys_dim)
            elif position == L:
                shape = (chil, *phys_dim)
            else:
                shape = (chil, chir, *phys_dim)
        else:
            if position == 1:
                shape = (1, chir, phys_dim[0])
            elif position == L:
                shape = (chil, 1, phys_dim[0])
            else:
                shape = (chil, chir, phys_dim[0])
    return shape
    
def SMPO_initialize(L: int,
            initializer: Initializer,
            key: Any,
            dtype: Any = jnp.float_,
            shape_method: str = 'even',
            spacing: int = 2,
            bond_dim: int = 4,
            phys_dim: Tuple[int, int] = (2, 2),
            output_inds: Collection = [],
            add_identity: bool = False,
            add_to_output: bool = False,
            boundary: str = 'obc',
            cyclic: bool = False,
            compress: bool = False,
            insert: int = None,
            canonical_center: int = None,
            **kwargs) -> SpacedMatrixProductOperator:
    
    """Generates :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

    Parameters
    ----------
    L : int
        Number of tensors.
    initializer : :class:`jax.nn.initializers.Initializer`
        Type of tensor initialization function.
    key : Any
        Argument key is a PRNG key (e.g. from jax.random.key()), used to generate random numbers to initialize the array.
    dtype : Any
        Type of tensor data (from `jax.numpy.float_`)
    shape_method : str
        Method to generate shapes for tensors.
    spacing : int
        Spacing paramater, or space between output indices in number of sites. When spacing is even.
    bond_dim : int
        Dimension of virtual indices between tensors. *Default = 4*.
    phys_dim :  tuple(int, int)
        Dimension of physical indices for individual tensor - *up* and *down*.
    output_inds : array of int
        Indexes of tensors which have output indices. From 0 to n. If spacing is not even.
    add_identity : bool
        Flag for adding identity to tensor diagonal elements. *Default=False*.
    add_to_output : bool
        Flag for adding identity to diagonal elements of tensors with output indices. *Default=False*.
    boundary : str
        Boundary condition. *Default='obc'*. obc = open boundary conditions, pbc = periodic boundary conditions.
    cyclic : bool
        Flag for indicating if SpacedMatrixProductOperator is cyclic. *Default=False*.
    compress : bool
        Flag to truncate bond dimensions.
    insert : int
        Index of tensor divided by norm. When `None` the norm division is distributed across all tensors
    canonical_center : int
        If not `None` then create canonical form around canonical center index.

    Returns
    -------
    :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
    """
    
    if cyclic and shape_method != 'even':
        raise NotImplementedError("Change shape_method to 'even'.")

    if 0 not in output_inds and len(output_inds) != 0:
        raise ValueError("First tensor needs to have output index.")

    if spacing == 1:
        raise ValueError("Spacing must be > 1, otherwise is Matrix Product Operator.")
    
    if initializer is not None and callable(initializer) and 'rand_unitary' in initializer.__qualname__:
        if add_identity:
            raise ValueError("rand_unitary initializer does not support add_identity.")
        if compress:
            raise ValueError("rand_unitary initializer does not support compress.")
        if insert:
            raise ValueError("rand_unitary initializer does not support insert.")
        if canonical_center:
            raise ValueError("rand_unitary initializer does not support canonization.")
        if boundary == 'obc':
                boundary = None

    if output_inds:
        hasoutput = []
        for i in range(L):
            if i in output_inds: hasoutput.append(True)
            else: hasoutput.append(False)
        spacings = [(o - output_inds[i]) for i, o in enumerate(output_inds[1:])]
        if (L - 1 - output_inds[-1]) != 0:
            spacings.append(L - 1 - output_inds[-1])
    else:
        hasoutput = itertools.cycle([True, *[False] * (spacing - 1)])

    tensors = []; out_index = 0
    for i, has_out in zip(range(1, L + 1), hasoutput):
        
        if output_inds:
            if len(spacings) - 1 >= out_index:
                spacing = spacings[out_index]
            if has_out:
                out_index+=1

        shape = generate_shape(shape_method, L, has_out, bond_dim, phys_dim, cyclic, i, spacing)

        tensor = initializer(key, shape, dtype)

        if add_identity:
            if len(tensor.shape) == 3:
                copy_tensor = jnp.copy(tensor)
                copy_tensor.at[:, :, 0].set(jnp.eye(tensor.shape[0],
                                                tensor.shape[1],
                                                dtype=dtype))
                tensor = copy_tensor
            elif len(tensor.shape) == 4: # output node
                if add_to_output:
                    copy_tensor = jnp.copy(tensor)
                    identity = jnp.eye(tensor.shape[0],
                                    tensor.shape[1],
                                    dtype=dtype)
                    identity = jnp.expand_dims(identity, axis=2)
                    identity = jnp.broadcast_to(identity, (copy_tensor.shape[0], copy_tensor.shape[1], copy_tensor.shape[3]))
                    copy_tensor.at[:, :, 0, :].set(identity)
                    tensor = copy_tensor
                
        if boundary == 'obc':
            aux_tensor = jnp.zeros(tensor.shape, dtype=dtype)
            if len(tensor.shape) == 3:
                if i == 1:
                    # Left node
                    aux_tensor = aux_tensor.at[:,0,:].set(tensor[:,0,:])
                    tensor = aux_tensor
                elif i == L:
                    # Right node
                    aux_tensor = aux_tensor.at[0,:,:].set(tensor[0,:,:])
                    tensor = aux_tensor
            elif len(tensor.shape) == 4:
                if i == 1:
                    # Left node
                    aux_tensor = aux_tensor.at[:,0,:,:].set(tensor[:,0,:,:])
                    tensor = aux_tensor
                elif i == L:
                    # Right node
                    aux_tensor = aux_tensor.at[0,:,:,:].set(tensor[0,:,:,:])
                    tensor = aux_tensor
        tensors.append(jnp.squeeze(tensor)/jnp.linalg.norm(tensor))
    
    if insert and insert < L and shape_method == 'even':
        tensors[insert] /= np.sqrt(min(bond_dim, phys_dim[0]))
    
    smpo = SpacedMatrixProductOperator(tensors, output_inds=output_inds, **kwargs)

    if compress:
        smpo.compress(form="flat", max_bond=bond_dim)  # limit bond_dim

    if L > 200:  # for large systems
        for i, tensor in enumerate(smpo.tensors):
            if i == 0:
                smpo.left_canonize_site(i)
            elif i == L - 1:
                tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
            else:
                tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
                smpo.left_canonize_site(i)
        
        if canonical_center is not None:
            smpo.canonicalize(canonical_center, inplace=True)
            smpo.normalize(insert=canonical_center, output_inds=output_inds)
        
    else:
        if canonical_center is None:
            smpo.normalize(output_inds=output_inds)
        else:
            smpo.canonicalize(canonical_center, inplace=True)
            smpo.normalize(insert=canonical_center, output_inds=output_inds)
    
    return smpo