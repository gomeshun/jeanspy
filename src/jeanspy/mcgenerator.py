import pandas as pd
import numpy as np
from  copy import deepcopy
import time 
from scipy.stats import chi2

def metropolis_generator(num_iter,logpdf_func,args_logpdf,logpdf_value,dargs_logpdf,push_time=10,proposal_dist="normal",print_func=print):
    '''
    E#xplanation: calculate next logpdf by metropolis algorithm
    arguments:
        logpdf_value = logpdf_func(**args_logpdf)
        args_logpdf,dargs_logpdf: pd.Series, same index
    '''
    proposal_dist_dic = {"normal":np.random.normal,"multivariate_normal":np.random.multivariate_normal}
    proposal_dist_func = proposal_dist_dic[proposal_dist]
    
    # initialization
    _args_logpdf = deepcopy(args_logpdf)
    _logpdf_value = logpdf_value
    _num_iter = 0
    t = time.time()
    num_accepted = 0

    while _num_iter<num_iter:
        args_logpdf_candidate = pd.Series(
            data = proposal_dist_func(_args_logpdf,dargs_logpdf),
            index = _args_logpdf.index
            )
        logpdf_value_candidate = logpdf_func(**args_logpdf_candidate)
        
        acceptance_rate = np.exp(logpdf_value_candidate-_logpdf_value)
        
        if np.isnan(acceptance_rate):
            mes = "NaN in acceptance rate calculation at " + str(_num_iter) + "th calculation \n"
            mes += str(_logpdf_value) + ' -> ' + str(logpdf_value_candidate) + '\n'
            mes += "args_logpdf:\n" + str(_args_logpdf) + '\n'
            mes += "args_logpdf_candidate:\n" + str(args_logpdf_candidate)
            raise TypeError(mes)
        
        if np.random.rand() < acceptance_rate: # accepted
            yield {"args_logpdf":args_logpdf_candidate,"logpdf_value":logpdf_value_candidate}
            _args_logpdf = args_logpdf_candidate
            _logpdf_value = logpdf_value_candidate
            _num_iter += 1
            num_accepted += 1
        else:
            yield {"args_logpdf":_args_logpdf,"logpdf_value":_logpdf_value}
            _num_iter += 1
        
        # for next iteration
        dt = time.time() - t
        if dt > push_time:
            print(format(_num_iter/num_iter,".3%")+" completed... acceptance rate: "+ format(num_accepted/_num_iter,'.2%'))
            t = time.time()

class MCgenerator:
    def __init__(self,logpdf_func,args_logpdf_init,dargs_logpdf=None,proposal_dist="normal",print_func=print,**options):
        '''
        initialization of logpdfs and memory (DataFrame of Monte-Carlo chain)

        logpdf_func: callable object, logpdf_func(**args_logpdf_init)
        args_logpdf_init(pd.DataFrame): initial args of logpdf_func.
        dargs_logpdf(pd.DataFrame): stride of MCMC. It's index must be same as that of args_logpdf.
        **options: option key words. aveilables:
            push_time(default=10): interval of push notification of MCMC (unit:second)
        '''
        print("initialization of MCgenerator start.")
        self.dargs_logpdf_log = pd.Series()
        self.num_iter_log = []
        self.logpdf_func = logpdf_func
        self.proposal_dist = proposal_dist
        print("function loaded.")
        self.args_logpdf_init = pd.Series(args_logpdf_init)
        self.logpdf_value_init = self.logpdf_func(**self.args_logpdf_init)
        print("logpdf_initialization completed.")
        # initialization of chain
        self.args_logpdf_chain = pd.DataFrame([self.args_logpdf_init,])
        self.logpdf_value_chain = [self.logpdf_value_init,]
        print("Data chains are initialized.")
        # initialization of MC method
        self.dargs_logpdf = {}
        self.options = {}
        self.print_func = print_func
        if dargs_logpdf is not None:
            self.update_MCparameter(dargs_logpdf,**options)
            print("MCparameters are initialized.")
        self.update_options(**options)
   
    def __repr__(self):
        return self.to_DataFrame().__repr__()

    def update_MCparameter(self,dargs_logpdf,print_mes=True):
        if print_mes:
            print("update dargs_logpdf:")
            print("before:")
            self.print_func(self.dargs_logpdf)
        self.dargs_logpdf = (dargs_logpdf if type(dargs_logpdf) != dict else pd.Series(dargs_logpdf) )
        self.nparams = len(dargs_logpdf)
        if print_mes:
            print("after:")
            self.print_func(self.dargs_logpdf)
        #if np.any(self.dargs_logpdf.index != self.args_logpdf_init.index):
        #    raise TypeError("parameters are not coincide between args and dargs of logpdf")
        
    def update_options(self,**options):
        self.options = options

    def generate(self,num_iter,with_message=True,proposal_dist=None):
        '''
        generate MC_chain and store them.
        num_iter(int): length of MC chain
        '''
        if proposal_dist==None:
            proposal_dist = self.proposal_dist
        results = [np.nan]*num_iter
        gen = metropolis_generator(
            num_iter = num_iter,
            logpdf_func = self.logpdf_func,
            args_logpdf = self.args_logpdf_chain.iloc[-1],
            logpdf_value = self.logpdf_value_chain[-1],
            dargs_logpdf = self.dargs_logpdf,
            proposal_dist = proposal_dist,
            **self.options
            )
        mes = "MCgeneration start.\noptions: " + str(self.options)
        if with_message:
            print(mes)
        results = [res for res in gen] 
        #for i in range(num_iter-1):
        #    results[0] = metropolis(
        #        logpdf_func = self.logpdf_func,
        #        args_logpdf = self.args_logpdf_chain.iloc[-1],
        #        logpdf_value = self.logpdf_value_chain[-1],
        #        dargs_logpdf = self.dargs_logpdf
        #        )
        if with_message:
            print("MCgeneration end.")
        results_args_logpdf = [result["args_logpdf"] for result in results]
        results_logpdf_value = [result["logpdf_value"] for result in results]
        self.args_logpdf_chain = self.args_logpdf_chain.append(results_args_logpdf,ignore_index=True)
        self.logpdf_value_chain.extend(results_logpdf_value)
        if with_message:
            print("MCresults are stored.")
        # logging
        #print(self.dargs_logpdf_log,"\n",self.dargs_logpdf)
        self.dargs_logpdf_log = self.dargs_logpdf_log.append(self.dargs_logpdf)
        #print(self.dargs_logpdf_log)
        self.num_iter_log.append(num_iter)
        
        if with_message:
            print("MCinfo are logged.")
            self.print_func(self.to_DataFrame(output_log=True))

    def generate_tuned(self,num_iter_list,repeat=1,proposal_dist=None,**kwargs):
        '''
        perform multi MCMC, updating darg_logpdf.
        
        num_iter_list: int, or list(int)
            length of each MCMC chain. if "int", repeat ("int") length chain by "repeat".
        repeat: repeat times of MCMC chain. If num_iter_list is "list", this is ignored.
        '''
        if proposal_dist==None:
            proposal_dist = self.proposal_dist
        def single_run(num_iter,isupdate_MCparameter=True,alpha=0.05,**_kwargs):
            '''
            function for single run.
                1. run the MCMC.
                2. update dargs_logpdf by the (co)variance of the MCMCed parameters
            '''
            # MCMC
            self.generate(num_iter,with_message=True,proposal_dist=self.proposal_dist)
            
            # update dargs_logpdf
            if isupdate_MCparameter:
                print(_kwargs["mes"]) if "mes" in _kwargs else None
                self.proposal_dist = proposal_dist
                last_args_logpdf_chain = self.args_logpdf_chain[-num_iter:]
                new_dargs_logpdf = {
                    "normal":np.sqrt(2.38**2/self.nparams)*last_args_logpdf_chain.std(),
                    "multivariate_normal":(2.38**2/self.nparams)*last_args_logpdf_chain.cov()
                }
                self.update_MCparameter(dargs_logpdf=new_dargs_logpdf[self.proposal_dist])
                '''
                dof = len(self.args_logpdf_chain)-1
                # check the latest variance by "args_logpdf_chain[-num_iter:]" (takes latest num_iter items)
                # NOTE: chi2 test, but here we perform multi chi2 test (#(parameters)).
                #       Is is appropriate ?
                _chi2 = dof*self.args_logpdf_chain[-num_iter:].var()/self.dargs_logpdf
                if np.any(np.logical_or(_chi2 < chi2.ppf(alpha/2,dof), chi2.ppf(1-alpha/2,dof) < _chi2)):
                    mes = "" if "i" not in kwargs else str(kwargs["i"])+"th iteration: "
                    mes += "update dargs_logpdf. \nbefore:\n"+ str(self.dargs_logpdf) + "\n"
                    self.update_MCparameter(dargs_logpdf=self.args_logpdf_chain[-num_iter:].std())
                    mes += "after:\n" + str(self.dargs_logpdf)
                    print(mes)
                '''

        if type(num_iter_list) == int:
            _num_iter_list = [num_iter_list,]*repeat
        else:
            _num_iter_list = num_iter_list
            repeat = len(_num_iter_list)
        
        [ single_run(num_iter,isupdate_MCparameter=(i<repeat),mes="{}th itetation".format(i),**kwargs) for num_iter,i in zip(_num_iter_list,range(repeat)) ]

    def to_DataFrame(self,output_log = False):
        '''
        output DataFrame.
        '''
        if not output_log:
            df = self.args_logpdf_chain.copy()
            df['logpdf'] = self.logpdf_value_chain
        else:
            df = self.dargs_logpdf_log.copy()
            #df['iter_num'] = self.num_iter_log
        return df

    def to_csv(self,filename,output_log = False):
        '''
        output csv file.
        NOTE: index of output file is None, so use to_DataFrame().to_csv(**kwargs) if you want indexed csv.
        '''
        df = self.to_DataFrame(output_log)
        df.to_csv(filename,index=None)


if __name__ == "__main__":
    logpdf = lambda x: -x*x/2
    gen = MCgenerator(logpdf,args_logpdf_init={"x":0},dargs_logpdf={"x":10})
    gen.generate_tuned([1000,]*5)
    #print(gen.args_logpdf_chain)
    #print(gen.logpdf_value_chain)
    from plot_setup import *
    gen.args_logpdf_chain.hist(bins=64)
    plt.show()
    plt.plot(range(5001),gen.logpdf_value_chain)
    plt.show()
    print(gen.to_DataFrame(output_log=True))
    input("press any key")

