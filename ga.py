from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
import torch
import pandas as pd
import numpy as np

class RocksDBSingleProblem(Problem):
    def __init__(self, knobs, model):
        self.knobs = knobs
        self.model = model
        self.model.eval()
        n_var = len(self.knobs.columns)
        n_obj = 1
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(x)).cuda()
        
        outputs = self.model(x)
        outputs = self.single_score_function(self.knobs.default_trg_em, outputs.cpu().numpy())
        out["F"] = outputs
        
    def single_score_function(df, pr):
        if df.size > 1:
            df = df.squeeze()
        score = (df[0] - pr[0]) + (pr[1] - df[1]) + (df[2] - pr[2]) + (df[3] - pr[3])
        return round(-score, 6)

class RocksDBMultiProblem(Problem):
    def __init__(self, knobs, model):
        self.knobs = knobs
        self.model = model
        self.model.eval()
        n_var = len(self.knobs.columns)
        n_obj = self.knobs.default_trg_em.shape[-1] # # of external metrics
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(x)).cuda()
        
        outputs = self.model(x)
        outputs = np.round(self.addb.scaler_y.inverse_transform(outputs.cpu().detach().numpy()), 2)
        out["F"] = outputs
        
def genetic_algorithm(mode, problem, pop_size, eliminate_duplicates=True):
    if mode == 'GA':
        algorithm = GA(pop_size=pop_size, eliminate_duplicates=eliminate_duplicates)
    elif mode == 'NSGA2':
        algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=eliminate_duplicates)
    res = minimize(problem, algorithm, verbose=False)
    return res