import itertools
import logging
import os
from multiprocessing import Pool, cpu_count

import numpy as np

try:
    import swyft  # type: ignore
except Exception as exc:
    swyft = None
    _SWYFT_IMPORT_ERROR = exc
else:
    _SWYFT_IMPORT_ERROR = None

try:
    import torch  # type: ignore
except Exception as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


def _raise_swyft_unavailable():
    message = [
        "swyft legacy features are unavailable in this environment.",
        "Install compatible swyft and torch dependencies to use DSphSimulator/Network.",
    ]
    if _SWYFT_IMPORT_ERROR is not None:
        message.append(f"swyft import failed: {_SWYFT_IMPORT_ERROR}")
    if _TORCH_IMPORT_ERROR is not None:
        message.append(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    raise ImportError(" ".join(message)) from (_SWYFT_IMPORT_ERROR or _TORCH_IMPORT_ERROR)


class DSphSimulator(swyft.Simulator if swyft is not None else object):
    """Legacy simulator backed by swyft."""

    def __init__(self, model):
        if swyft is None:
            _raise_swyft_unavailable()
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.model = model
        self.logger = logger.getChild(self.__class__.__name__)
        self.R_pc = self.model.data["R_pc"].values
        self.dsph_model = self.model.submodels["DSphModel"]
        self.vmem_kms_index = self.model.submodels["FlatPriorModel"].data.index.get_loc("vmem_kms")
        self.store_dir = "DSphSimulator_samples"
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)
        self.store = {}

    def printlog(self, *txt):
        self.logger.info("%s", " ".join(map(str, txt)))

    def get_store_path(self, fname):
        return os.path.join(self.store_dir, fname)

    def get_vmem_kms(self, p):
        return p[self.vmem_kms_index]

    def get_slos(self, p):
        params = self.model.convert_params(p)
        self.model.update("all", params)
        return self.dsph_model.sigmalos_dequad(self.R_pc)

    def get_vlos_kms(self, vmem_kms, slos):
        return vmem_kms + slos * np.random.randn(len(self.R_pc))

    def build(self, graph):
        p = graph.node("p", self.model.sample)
        vmem_kms = graph.node("vmem_kms", self.get_vmem_kms, p)
        slos = graph.node("slos_kms", self.get_slos, p)
        graph.node("vlos_kms", self.get_vlos_kms, vmem_kms, slos)

    @property
    def default_zarr_filename(self):
        return f"zarr_{self.model.__class__.__name__}_{self.model.dsph_name}"

    def init_zarrstore(self, N, chunk_size, targets=None, zarr_filename=None):
        if targets is None:
            targets = {}
        zarr_filename = self.default_zarr_filename if zarr_filename is None else zarr_filename
        self.store[zarr_filename] = swyft.ZarrStore(self.get_store_path(zarr_filename))
        shapes, dtypes = self.get_shapes_and_dtypes(targets)
        self.store[zarr_filename].init(N, chunk_size, shapes, dtypes)

    def simulate_in_zarrstore(
        self,
        N,
        chunk_size,
        batch_size,
        targets=None,
        conditions=None,
        exclude=None,
        parallel=True,
        zarr_filename=None,
    ):
        if conditions is None:
            conditions = {}
        if exclude is None:
            exclude = []
        self.init_zarrstore(N, chunk_size, targets, zarr_filename)

        worker = Worker(self, targets, conditions, exclude)
        store = self.store[zarr_filename]

        if parallel:
            self.printlog("Parallel sampling started.")
            n_pool = min(N // batch_size, cpu_count())
            self.printlog("Parallel sampling with", n_pool, "processes.")
            with Pool(n_pool) as pool:
                pool.starmap(store.simulate, itertools.repeat([worker, None, batch_size], n_pool))
        else:
            store.simulate(worker, batch_size)


class Worker:
    """Wrapper passed to swyft.ZarrStore.simulate."""

    def __init__(self, sampler, targets, conditions, exclude):
        self.sampler = sampler
        self.targets = targets
        self.conditions = conditions
        self.exclude = exclude

    def __call__(self, size):
        return self.sampler.sample(size, self.targets, self.conditions, self.exclude)


class Network(swyft.SwyftModule if swyft is not None else object):
    def __init__(self, model, enable_1dim=True, enable_ndim=False, enable_embedding=False, dropout=0.1):
        if swyft is None or torch is None:
            _raise_swyft_unavailable()

        assert enable_1dim or enable_ndim

        super().__init__()
        self.model = model
        self.ndim = model.ndim
        self.enable_1dim = enable_1dim
        self.enable_ndim = enable_ndim
        self.enable_embedding = enable_embedding
        self.dropout = dropout

        if enable_embedding:
            n_embedding = model.ndim
            self.embedding = torch.nn.Linear(len(model.data), n_embedding)

        num_features = self.n_embedding if enable_embedding else len(model.data)
        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=num_features,
            num_params=model.ndim,
            varnames="p",
            dropout=dropout,
        )

        if enable_ndim:
            self.marginals = [(0, 1)]
            self.logratios2 = swyft.LogRatioEstimator_Ndim(
                num_features=num_features,
                marginals=self.marginals,
                varnames="p",
                dropout=dropout,
            )

    def forward(self, A, B):
        A = self.embedding(A["vlos_kms"]) if self.enable_embedding else A["vlos_kms"]

        logratios = []
        if self.enable_1dim:
            logratios.append(self.logratios1(A, B["p"]))
        if self.enable_ndim:
            logratios.append(self.logratios2(A, B["p"]))

        return logratios


__all__ = ["DSphSimulator", "Network", "Worker", "_raise_swyft_unavailable"]