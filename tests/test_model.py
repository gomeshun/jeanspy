# unit test for model.py
import unittest
import os
import tempfile

import pytest

from jeanspy.model import *
import jeanspy
import numpy as np
from scipy.stats import norm, uniform
from jeanspy.sampler import *
swyft = None
import matplotlib.pyplot as plt

torch = None
pl_loggers = None

try:
    import torch as _torch
    from pytorch_lightning import loggers as _pl_loggers
except Exception:
    _LEGACY_SWYFT_REASON = (
        "legacy swyft/torch tests are disabled unless the optional swyft, torch, and "
        "pytorch_lightning stack is installed"
    )
else:
    torch = _torch
    pl_loggers = _pl_loggers
    _LEGACY_SWYFT_REASON = "swyft is temporarily disabled in this codebase"


pytestmark = pytest.mark.skip(reason=_LEGACY_SWYFT_REASON)


def _skip_if_swyft_unavailable(testcase: unittest.TestCase):
    missing = []
    if swyft is None:
        missing.append("swyft")
    if torch is None:
        missing.append("torch")
    if pl_loggers is None:
        missing.append("pytorch_lightning")
    if missing:
        testcase.skipTest(
            "legacy swyft/torch tests disabled because optional dependencies are unavailable: "
            + ", ".join(missing)
        )


class TestModel(unittest.TestCase):
    def setUp(self):
        _skip_if_swyft_unavailable(self)

    def test_Model(self):
        """ test all features of the Model class 
        """
        class MyModel1(Model):
            """ Model class which has two parameters. 
            """
            required_param_names = ['a','b']
            required_models = {}

            def return_a(self):
                return self.params['a']
            
            def return_b(self):
                return self.params['b']
                

        class MyModel2(Model):
            """ Model class which has two parameters.
            """
            required_param_names = ['c','d']
            required_models = {}

            def return_c(self):
                return self.params['c']

            def return_d(self):
                return self.params['d']

        
        class MyModel1and2(Model):
            """ Model class which has two models as its submodels.
            """
            required_param_names = ['x','y']
            required_models = {'m1':MyModel1,'m2':MyModel2}

            def return_x(self):
                return self.params['x']
            
            def return_y(self):
                return self.params['y']

            def return_m1a(self):
                return self.submodels['m1'].return_a()

            def return_m1b(self):
                return self.submodels['m1'].return_b()

        # check that we can properly create a model
        a0 = 1
        b0 = 2
        c0 = 3
        d0 = 4
        x0 = 5
        y0 = 6

        mdl1 = MyModel1(a=a0,b=b0)
        mdl2 = MyModel2(c=c0,d=d0)
        mdl1and2 = MyModel1and2(x=x0,y=y0,submodels={'m1':mdl1,'m2':mdl2})

        # check that we can properly update a model 
        x1 = 10
        y1 = 20
        a1 = 30
        b1 = 40
        c1 = 50
        d1 = 60
        mdl1and2.update(x=x1,y=y1)
        mdl1and2.submodels['m1'].update(a=a1,b=b1)
        mdl1and2.submodels['m2'].update(c=c1,d=d1)
        
        # check that we can properly get a model's parameters
        self.assertEqual(mdl1and2.params['x'],x1)
        self.assertEqual(mdl1and2.params['y'],y1)
        self.assertEqual(mdl1and2.submodels['m1'].params['a'],a1)
        self.assertEqual(mdl1and2.submodels['m1'].params['b'],b1)
        self.assertEqual(mdl1and2.submodels['m2'].params['c'],c1)
        self.assertEqual(mdl1and2.submodels['m2'].params['d'],d1)

        # check that we can update sumbodel parameters from the top level
        a2 = 300
        b2 = 400
        mdl1and2.update(a=a2,b=b2)

        # chech that we can properly get a model's parameters
        self.assertEqual(mdl1and2.submodels['m1'].params['a'],a2)
        self.assertEqual(mdl1and2.submodels['m1'].params['b'],b2)


        # check that the model raises error when invalid parameter name is given
        with self.assertRaises(ValueError):
            mdl1and2.update(z=0)

    def test_FittableModel_from_scratch(self):
        self._test_FittableModel()

    def test_FittableModel_from_checkpoint(self):
        checkpoint_path = "lightning_logs/FittableModel1_SimpleGaussianModel/version_18/checkpoints/epoch=199-step=16000.ckpt"
        self._test_FittableModel(checkpoint_path=checkpoint_path)

    def _test_FittableModel(self,checkpoint_path=None):
        """ test all features of the FittableModel class """

        class SimpleGaussianModel(Model):
            required_param_names = ['a', 'b' ,'scale']
            required_models = {}

            def y_pred(self,x):
                """ return prediction value of y = a * x + b """
                return self.params['a'] * x + self.params['b']


        class FittableModel1(FittableModel):
            """ FittableModel class which has three parameters, 
            describing a simple gaussian model.
            """
            required_param_names = []
            required_models = {'Model1':SimpleGaussianModel}
            prior_names = ["Prior1"]
            
            def convert_params(self, p):
                """ return parameters.
                p : [ a, b, log(scale) ]
                return: { "a" : a, "b" : b ,"scale": 10**scale }
                """
                return pd.Series({"a": p[0], "b" : p[1],  "scale": 10.0 ** p[2]})

            def load_data(self, data):
                """ load data. data has column y """
                self.data = data
            

            def _lnlikelihoods(self, *args, **kwargs):
                """ return log likelihoods """
                return norm.logpdf(self.data["y"], 
                                   loc=self.submodel["Model1"].y_pred(self.data["x"]), 
                                   scale=self.params['scale'])

            def _lnpriors(self):
                return 0

        # check fitting with a simple gaussian model

        class TestSimulator(swyft.Simulator):
            def __init__(self,model):
                super().__init__()
                self.transform_samples = swyft.to_numpy32
                self.model = model

            def get_y(self,p):
                params = self.model.convert_params(p)
                self.model.update("all",params)
                y = self.model.submodels["Model1"].y_pred(self.model.data["x"].values)
                return y

            def build(self, graph):
                node_p = graph.node("p", lambda: np.random.uniform([-100,-100,-10],[+100,+100,+10]))
                node_x = graph.node("y", lambda p: self.get_y(p), node_p)
        

        class Network(swyft.SwyftModule):
            def __init__(self,model):
                super().__init__()
                # self.embedding = torch.nn.Linear(len(model.data), model.ndim)
                # print(self.embedding)
                self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = len(model.data), num_params = model.ndim, varnames = 'p')
                # NOTE: Each marginal in marginals must be ascending order
                self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = len(model.data), marginals = [[0,1],[1,2],[0,2]], varnames = 'p')

            def forward(self, A, B):
                logratios1 = self.logratios1(A['y'], B['p'])
                logratios2 = self.logratios2(A['y'], B['p'])
                # logratios2 = self.logratios2(embedding, B['p'])
                return logratios1, logratios2
            
        # if checkpoint_path is None, we train the network from scratch:
        
        data = pd.DataFrame({"x":np.linspace(-10,10).astype(np.float32)})
        mdl = FittableModel1(args_load_data = [data],submodels={
                "Model1":SimpleGaussianModel(),
            })
            
        network = Network(mdl)

        # prepare the data and datamodule
        simulator = TestSimulator(mdl)
        store = swyft.ZarrStore(f"zarr_{mdl.name}")
        shapes, dtypes = simulator.get_shapes_and_dtypes()
        store.init(10000, 100, shapes, dtypes)
        store.simulate(simulator, batch_size=100)  # if already generated, this will do nothing
        dm = swyft.SwyftDataModule(store, fractions=[0.8,0.1,0.1],batch_size=100,num_workers=3)
            

        if checkpoint_path is None:
            # prepare the datamodule and trainer
            logger = pl_loggers.TensorBoardLogger('lightning_logs/', name=mdl.name)
            trainer = swyft.SwyftTrainer(accelerator="gpu",max_epochs=200,logger=logger)
            
            # fit the network
            trainer.fit(network, dm)
        
        else:
            # Instead of fitting, load the pretrained network
            network.load_state_dict(torch.load(checkpoint_path)["state_dict"])
            trainer = swyft.SwyftTrainer(accelerator="gpu")

        # test the network
        trainer.test(network,dm)

        # check the coverage
        B = store[:1000]
        A = store[:1000]
        mass = trainer.test_coverage(network, A, B)
        print(mass)

        # make directory
        os.makedirs(mdl.name, exist_ok=True)

        # check PP-plot
        for i,_parname in enumerate(mass[0].parnames):
            parname = _parname[0]
            fig,ax = plt.subplots()
            swyft.plot_pp(mass,parname,ax=ax)
            ax.set_title(parname)
            # save figure
            fig.savefig(f"{mdl.name}/test_pp_{parname}.png")

        # check ZZ-plot
        for i,_parname in enumerate(mass[0].parnames):
            parname = _parname[0]
            fig,ax = plt.subplots()
            swyft.plot_zz(mass,parname,ax=ax)
            ax.set_title(parname)
            # save figure
            fig.savefig(f"{mdl.name}/test_zz_{parname}.png")

        # inferrence 
        A = simulator.sample(targets=["y"], 
                             conditions={"p":np.array([1,2,-1])})
        B = simulator.sample(100000,targets=["p"])
        print(A)
        print(B)

        # prediction
        predictions = trainer.infer(network,A,B)
        for _parname in predictions[0].parnames:
            parname = _parname[0]
            fig,ax = plt.subplots()
            swyft.plot_1d(predictions,parname,ax=ax)
            ax.set_title(parname)
            # save figure
            fig.savefig(f"{mdl.name}/test_inference_{parname}.png")

        swyft.corner(predictions, ('p[0]', 'p[1]', 'p[2]'), bins = 200, smooth = 3)
        plt.savefig(f"{mdl.name}/test_inference_corner.png")


class TestDSphSimulator(unittest.TestCase):
    """ Test DSphSimulator
    """
    
    
    def setUp(self) -> None:
        _skip_if_swyft_unavailable(self)
        """ prepare .test directory. 
        """
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """ remove .test directory. 
        """
        self.temp_dir.cleanup()

    def test_sample(self):
        dsph_name = "Fornax"
        mdl = get_default_estimation_model(dsph_name,"Classical")
        sim = DSphSimulator(model=mdl)
        
        vmem_kms = 100
        log10_re_pc = 2
        log10_rs_pc = 3
        log10_rhos_Msunpc3 = 1
        log10_r_t_pc = 4
        bfunc_beta_ani = 0.1
        p = [vmem_kms,log10_re_pc,log10_rs_pc,log10_rhos_Msunpc3,log10_r_t_pc,bfunc_beta_ani]
        params = mdl.convert_params(p)
        
        sample = sim.sample(1000,targets=["vlos_kms"],conditions={"p":p})
        slos_obs = np.std(sample["vlos_kms"],axis=0,ddof=1)
        mdl.update("all",params)
        R_pc = mdl.data["R_pc"].values
        slos_th = mdl.submodels["DSphModel"].sigmalos_dequad(mdl.data["R_pc"].values)
        # sort R_pc and slos_th by R_pc
        R_pc_sorted,slos_th_sorted = zip(*sorted(zip(R_pc,slos_th)))
        plt.plot(R_pc_sorted,slos_th_sorted,label="theory")
        plt.plot(R_pc,slos_obs,".",label="obs")
        plt.legend()
        plt.xscale("log")
        plt.xlabel("R_pc")
        plt.ylabel("slos_kms")
        plt.savefig("test_compare_slos_th_obs.png")

    
    def test_parallel_sample(self):
        dsph_name = "Fornax"
        mdl = get_default_estimation_model("Classical",dsph_name)
        sim = DSphSimulator(model=mdl)
        
        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test")

        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_p",targets=["p"])

        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_vlos_kms",targets=["vlos_kms"])

        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_vlos_kms_conditioned",
                                  targets=["vlos_kms"],
                                  conditions={"p":np.array([0,0,0,0,0,0])})

    
    def test_nonparallel_sample(self):
        dsph_name = "Fornax"
        mdl = get_default_estimation_model(dsph_name,"Classical")
        sim = DSphSimulator(model=mdl)
        
        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_np",parallel=False)

        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_np_p",targets=["p"],parallel=False)

        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_np_vlos_kms",targets=["vlos_kms"],parallel=False)

        sim.simulate_in_zarrstore(1000,100,100,zarr_filename=self.temp_dir.name+"/test_np_vlos_kms_conditioned",parallel=False,
                                  targets=["vlos_kms"],
                                  conditions={"p":np.array([0,0,0,0,0,0])})


class TestDSphEstimationModel(unittest.TestCase):
    """ Test SimpleDSphModel
    """
    def setUp(self):
        _skip_if_swyft_unavailable(self)

    def test_convert_param(self):
        dsph_name = "Fornax"
        mdl = get_default_estimation_model("Classical",dsph_name)
        print(mdl)

        
        vmem_kms = 100
        log10_re_pc = 2
        log10_rs_pc = 3
        log10_rhos_Msunpc3 = 1
        log10_r_t_pc = 4
        bfunc_beta_ani = 0.1
        p = [vmem_kms,log10_re_pc,log10_rs_pc,log10_rhos_Msunpc3,log10_r_t_pc,bfunc_beta_ani]
        params = mdl.convert_params(p)
        params_shouldbe = pd.Series(dict(
            vmem_kms = vmem_kms,
            re_pc = 10**log10_re_pc,
            rs_pc = 10**log10_rs_pc,
            rhos_Msunpc3 = 10**log10_rhos_Msunpc3,
            r_t_pc = 10**log10_r_t_pc,
            beta_ani = 1-10**bfunc_beta_ani,
        ))
        print(params)
        print(params_shouldbe)
        self.assertTrue(np.allclose(params,params_shouldbe))


    def test_SimpleDSphModel(self):
        mdl = jeanspy.model.DSphModel(submodels={
                "StellarModel" : jeanspy.model.PlummerModel(),
                "DMModel" : jeanspy.model.NFWModel(),
                "AnisotropyModel" : jeanspy.model.ConstantAnisotropyModel(),
            },vmem_kms=0)
        print(mdl)

        mdl.update(re_pc=200,rs_pc=2000,r_t_pc=20000,rhos_Msunpc3=1, beta_ani=0.5,vmem_kms=100)
        print(mdl)
        for beta_ani in [-3,-2,-1,-0.5,0,0.5,1]:
            mdl.update(beta_ani=beta_ani)
            R_pc = np.logspace(0,4)
            s2 = mdl.sigmalos2_dequad(R_pc)
            plt.plot(R_pc,np.sqrt(s2),label="beta_ani={}".format(beta_ani))
        plt.xscale("log")
        plt.legend()
        plt.savefig("test_SimpleDSphModel.png")

    
    def test_DSphEstimationModel_1dim_Mock(self):
        self._test_DSphEstimationModel(dsph_name="Mock",
                                       enable_1dim=True,
                                       enable_ndim=False,
                                       enable_embedding=False,
                                       batch_size=128,
                                       max_epochs=200,
                                       dropout=0.01)

    def test_DSphEstimationModel_1dim(self):
        self._test_DSphEstimationModel(dsph_name="Fornax",
                                       enable_1dim=True,
                                       enable_ndim=False,
                                       enable_embedding=False,
                                       batch_size=128,
                                       max_epochs=200,
                                       dropout=0.01)

    def test_DSphEstimationModel_ndim(self):
        self._test_DSphEstimationModel(dsph_name="Fornax",
                                       enable_1dim=False,
                                       enable_ndim=True,
                                       enable_embedding=False)

    def test_DSphEstimationModel_1dim_emb(self):
        self._test_DSphEstimationModel(dsph_name="Fornax",
                                       enable_1dim=True,
                                       enable_ndim=False,
                                       enable_embedding=True)
        
    def test_DSphEstimationModel_ndim_emb(self):
        self._test_DSphEstimationModel(dsph_name="Fornax",
                                       enable_1dim=False,
                                       enable_ndim=True,
                                       enable_embedding=True)
        
    
    def test_DSphEstimationModel_1andndim(self):
        self._test_DSphEstimationModel(dsph_name="Fornax",
                                       enable_1dim=True,
                                       enable_ndim=True,
                                       enable_embedding=False)
        

    def test_DSphEstimationModel_1andndim_emb(self):
        self._test_DSphEstimationModel(dsph_name="Fornax",
                                       enable_1dim=True,
                                       enable_ndim=True,
                                       enable_embedding=True)


    def test_DSphEstimationModel_from_checkpoint(self):
        checkpoint_path = "lightning_logs/SimpleDSphEstimationModel_DSphModel_PlummerModel+NFWModel+ConstantAnisotropyModel+FlatPriorModel+PhotometryPriorModel/version_0/checkpoints/epoch=98-step=12672.ckpt"
        self._test_DSphEstimationModel(checkpoint_path=checkpoint_path)


    def _test_DSphEstimationModel(self,
                                  dsph_name,
                                  enable_1dim,
                                  enable_ndim,
                                  enable_embedding,
                                  max_epochs=100,
                                  batch_size=64,
                                  checkpoint_path=None,
                                  dropout=0.1,
                                  precision=64):
        """ Test the DSphEstimationModel
        """

        # float32_matmul_precision = "high"
        # torch.set_float32_matmul_precision(float32_matmul_precision)

        # enable_1dim and enable_ndim cannot be False at the same time
        assert enable_1dim or enable_ndim

        # load model
        if dsph_name != "Mock":
            mdl = get_default_estimation_model("Classical",dsph_name)
            sim = jeanspy.sampler.DSphSimulator(mdl)
        else:
            # # load mock data from dsph_name file
            # df_mock = pd.read_csv(dsph_name)
            # dsph_name = pd.DataFrame({
            #     "R_pc" : dsph_name,
            #     "vlos_kms" : np.zeros_like(dsph_name)
            #     })
            # mdl = get_default_estimation_model("Mock",dsph_name)
            # sim = jeanspy.sampler.DSphSimulator(mdl)

            # In this case, we should set mdl.data manually 
            # and execute mdl.submodels["PriorModel"].reset_prior(loc,scale).
            mdl = get_default_estimation_model("Mock","Mock")
            R_pc = np.logspace(0,4,128)
            mdl.reset_data(pd.DataFrame({
                "R_pc" : R_pc,
                "vlos_kms" : np.zeros_like(R_pc)
            }))
            mdl.submodels["PhotometryPriorModel"].reset_prior(loc=2.0,scale=0.1)
            sim = jeanspy.sampler.DSphSimulator(mdl)

            # declare true model parameters and get mock data
            p_true = [
                0.0,    # vmem_kms,-1000,1000
                2.0,    # log10_re_pc,1.0,3.5
                3.0,    # log10_rs_pc,0,5
                0.0,    # log10_rhos_Msunpc3,-4,4
                4.0,    # log10_r_t_pc,0,5
                0.1     # bfunc_beta_ani,-1,1
            ]
            vlos_kms_obs = sim.sample(conditions = {"p":p_true},
                                      targets = ["vlos_kms"])["vlos_kms"]
            new_data = pd.DataFrame(mdl.data["vlos_kms"])
            new_data["vlos_kms"] = vlos_kms_obs
            mdl.reset_data(new_data)
            sim = jeanspy.sampler.DSphSimulator(mdl)
            
        # simulate data
        # store = swyft.ZarrStore(f"zarr_{self.__class__.__name__}_{dsph_name}")
        # shapes, dtypes = sim.get_shapes_and_dtypes()
        # store.init(10000, 100, shapes, dtypes)
        # # NOTE: The first argument must be an executable function taking a single integer argument representing the simulation size.
        # store.simulate(sim, batch_size = 1000)  # If already simulated, store.simulate do nothing
        zarr_filename = f"zarr_{self.__class__.__name__}_{dsph_name}"
        zarr_test_coverage_filename = f"zarr_test_coverage_{self.__class__.__name__}_{dsph_name}"
        zarr_inference_filename = f"zarr_inference_{self.__class__.__name__}_{dsph_name}"
        sim.simulate_in_zarrstore(20000,100,100,
                                  zarr_filename=zarr_filename)
        sim.simulate_in_zarrstore(1000,100,10,
                                  zarr_filename=zarr_test_coverage_filename)
        sim.simulate_in_zarrstore(1000000,10000,10000,targets=["p"],
                                  zarr_filename=zarr_inference_filename)
        
        

        # with Pool() as pool:
        #     worker = store.simulate
        #     n = # number of iteration
        #     pool.map(worker,n)

        network = Network(mdl,
                          enable_1dim=enable_1dim,
                          enable_ndim=enable_ndim,
                          enable_embedding=enable_embedding,
                          dropout=dropout)
        dm = swyft.SwyftDataModule(sim.store[zarr_filename], fractions=[0.8,0.2,0.0], batch_size=batch_size, num_workers=16)

        logger = pl_loggers.TensorBoardLogger("lightning_logs", name=f"{mdl.__class__.__name__}_{dsph_name}")
        trainer = swyft.SwyftTrainer(accelerator="gpu",max_epochs=max_epochs, logger=logger, precision=precision)

        # logging input paramters
        logger.experiment.add_text("dsph_name",dsph_name)
        logger.experiment.add_text("enable_emmbeding",str(enable_embedding))
        logger.experiment.add_text("enable_ndim",str(enable_ndim))
        logger.experiment.add_text("checkpoint_path",str(checkpoint_path))
        logger.experiment.add_text("model_name",str(mdl.name))
        logger.experiment.add_text("network",str(network))
        logger.experiment.add_text("max_epochs",str(max_epochs))
        logger.experiment.add_text("batch_size",str(batch_size))
        logger.experiment.add_text("dropout",str(dropout))
        logger.experiment.add_text("precision",str(precision))
        if enable_ndim:
            logger.experiment.add_text("margnials",str(network.marginals))
        logger.experiment.add_text("torch.float32_matmul_precision",torch.get_float32_matmul_precision())
        
        if checkpoint_path is None:
            # train the network
            trainer.fit(Network(mdl),dm,ckpt_path=checkpoint_path)
        else:
            network.load_state_dict(torch.load(checkpoint_path)["state_dict"])

        # # test the network
        # trainer.test(network,dm)

        # check the coverage
        # A = sim.store[zarr_filename][:10000]
        # load test dataset from DataModule "dm"
        A = sim.store[zarr_test_coverage_filename][:1000]
        B = sim.store[zarr_inference_filename][:10000]
        mass = trainer.test_coverage(network, A, B)  # Here len(mass) == 2
        
        # if enable_ndim:
        #     mass = mass[0]
        mass = mass[0]  # NOTE: mass is a list and mass[0] must be the output of logratio_1dim

        # make model directory
        os.makedirs(mdl.name,exist_ok=True)

        # check PP-plot
        for i,_parname in enumerate(mass.parnames):
            parname = _parname[0]
            fig,ax = plt.subplots()
            swyft.plot_pp(mass,parname,ax=ax)
            ax.set_title(parname)
            # save figure
            # fig.savefig(f"{mdl.name}/pp_{parname}.png")
            logger.experiment.add_figure(f"pp_plot",fig,i)

        # check ZZ-plot
        for i,_parname in enumerate(mass.parnames):
            parname = _parname[0]
            fig,ax = plt.subplots()
            swyft.plot_zz(mass,parname,ax=ax)
            ax.set_title(parname)
            # save figure
            # fig.savefig(f"{mdl.name}/zz_{parname}.png")
            logger.experiment.add_figure(f"zz_plot",fig,i)


        # inferrence 
        dsphdata = mdl.data
        A = swyft.Sample(vlos_kms = dsphdata["vlos_kms"].values.astype(np.float32))
        # obtain parameter sample from simulator 
        # store_p = swyft.ZarrStore("zarr_inference")
        # store_p.init(1000000,10000,*simulator.get_shapes_and_dtypes())
        # store_p.simulate(simulator,batch_size=10000)
        # B = sim.sample(1000000,targets=["p"])  # NOTE: Failed when 100000, maybe too small samples are bad for inference
        # NOTE: Instead using this method, we can use the following method
        B = sim.store[zarr_inference_filename][:1000000]
        
        # prediction
        predictions = trainer.infer(network,A,B)  # Here len(predictions) == 2
        # if enable_ndim:
        #     prediction = predictions[0]
        # else:
        #     prediction = predictions


        for i,_parname in enumerate(predictions[0].parnames):
            parname = _parname[0]
            fig,ax = plt.subplots()
            swyft.plot_1d(predictions[0],parname,ax=ax)
            ax.set_title(parname)
            # save figure
            # fig.savefig(f"{mdl.name}/inference_{parname}.png")
            logger.experiment.add_figure(f"inference_plot",fig,i)

        if enable_ndim:
            swyft.corner(predictions[1], [f'p[{i}]' for i in range(mdl.ndim)], bins = 200, smooth = 3)
            # plt.savefig(f"{mdl.name}/test_inference_corner.png")
            logger.experiment.add_figure(f"inference_corner_plot",plt.gcf())



if __name__ == '__main__':
    unittest.main()