from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
import torch
import pandas as pd
import numpy as np


class DBMSProblem():
    def __init__(self, knobs, model, opt):
        self.knobs = knobs
        self.model = model
        self.opt = opt
    
    def get_problem(self):
        if self.opt.dbms == 'rocksdb':
            if self.opt.ga == 'GA':
                return RocksDBSingleProblem(knobs=self.knobs, model=self.model, model_mode=self.opt.mode)
            elif self.opt.ga == 'NSGA2' or self.opt.ga == 'NSGA3':
                return RocksDBMultiProblem(knobs=self.knobs, model=self.model, model_mode=self.opt.mode)
        elif self.opt.dbms == 'mysql':
            if self.opt.ga == 'GA':
                return MySQLSingleProblem(knobs=self.knobs, model=self.model, model_mode=self.opt.mode)
            elif self.opt.ga == 'NSGA2' or self.opt.ga == 'NSGA3':
                return MySQLMultiProblem(knobs=self.knobs, model=self.model, model_mode=self.opt.mode)

class RocksDBSingleProblem(Problem):
    def __init__(self, knobs, model, model_mode):
        self.knobs = knobs
        self.model = model
        self.model_mode = model_mode
        # self.model.eval()
        n_var = len(self.knobs.columns)
        n_obj = 1
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(x)).cuda()
        
        if self.model_mode == "RF":
            outputs = self.model.predict(x.cpu().detach().numpy())
            outputs = self.single_score_function(self.knobs.default_trg_em, outputs)
        else:
            outputs = self.model(x)
            outputs = self.single_score_function(self.knobs.default_trg_em, outputs.cpu().detach().numpy())
        out["F"] = outputs
        
    def single_score_function(self, df, pr):
        df = np.repeat(df, pr.shape[0], axis=0)
        
        score = (df[:,0] - pr[:,0]) + (pr[:,1] - df[:,1]) + (df[:,2] - pr[:,2]) + (df[:,3] - pr[:,3])
        return np.round(-score, 6)

class RocksDBMultiProblem(Problem):
    def __init__(self, knobs, model, model_mode):
        self.knobs = knobs
        self.model = model
        self.model_mode = model_mode
        # self.model.eval()
        self.n_var = len(self.knobs.columns)
        n_obj = self.knobs.default_trg_em.shape[-1] # # of external metrics
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        
        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(pd.DataFrame(data=x, columns=self.knobs.columns))).cuda()
        
        if self.model_mode == "RF":
            outputs = self.model.predict(x.cpu().detach().numpy())
            # outputs = np.round(self.knobs.scaler_em.inverse_transform(outputs), 2)
            # outputs = np.round(self.knobs.scaler_em.inverse_transform(outputs), 2)
        else:
            outputs = self.model(x)
            outputs = outputs.cpu().detach().numpy()
            # outputs = np.round(self.knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
        outputs[:,1] = -outputs[:,1]
        out["F"] = outputs

class MySQLSingleProblem(Problem):
    def __init__(self, knobs, model, model_mode):
        self.knobs = knobs
        self.model = model
        self.model_mode = model_mode
        # self.model.eval()
        n_var = len(self.knobs.columns)
        n_obj = 1
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(x)).cuda()
        
        if self.model_mode == "RF":
            outputs = self.model.predict(x.cpu().detach().numpy())
            out["F"] = outputs[:,1] / outputs[:,0]
        else:
            outputs = self.model(x)
            out["F"] = outputs[:,1] / outputs[:,0]
        out["F"] = outputs

class MySQLMultiProblem(Problem):
    def __init__(self, knobs, model, model_mode):
        self.knobs = knobs
        self.model = model
        self.model_mode = model_mode
        # self.model.eval()
        self.n_var = len(self.knobs.columns)
        n_obj = self.knobs.default_trg_em.shape[-1] # # of external metrics
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        
        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(pd.DataFrame(data=x, columns=self.knobs.columns))).cuda()
        
        if self.model_mode == "RF":
            outputs = self.model.predict(x.cpu().detach().numpy())
            # outputs = np.round(self.knobs.scaler_em.inverse_transform(outputs), 2)
            # outputs = np.round(self.knobs.scaler_em.inverse_transform(outputs), 2)
        else:
            outputs = self.model(x)
            outputs = outputs.cpu().detach().numpy()
            # outputs = np.round(self.knobs.scaler_em.inverse_transform(outputs.cpu().detach().numpy()), 2)
        outputs[:,1] = -outputs[:,0]
        out["F"] = outputs

def genetic_algorithm(mode, problem, pop_size, eliminate_duplicates=True):
    if mode == 'GA':
        algorithm = GA(pop_size=pop_size, eliminate_duplicates=eliminate_duplicates)
    elif mode == 'NSGA2':
        algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=eliminate_duplicates)
    elif mode == 'NSGA3':
        assert False, "Too much memory required"
        ref_dirs = get_reference_directions('das-dennis', problem.n_var, n_partitions=problem.n_var*4)
        algorithm = NSGA3(pop_size=pop_size, eliminate_duplicates=eliminate_duplicates, ref_dirs=ref_dirs)
    res = minimize(problem, algorithm, verbose=False)
    return res