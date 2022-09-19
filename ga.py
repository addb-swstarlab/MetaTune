from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
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
#         dbms1 = torch.Tensor(addb.scaler_redis.transform(x[:, :self.addb.redis_len])).cuda()
#         dbms2 = torch.Tensor(addb.scaler_rocksdb.transform(x[:, self.addb.redis_len:self.addb.redis_len+self.addb.rocksdb_len])).cuda()
#         dbms3 = torch.Tensor(addb.scaler_spark.transform(x[:, -self.addb.spark_len:])).cuda()
        dbms1 = torch.Tensor(self.addb.scaler_redis.transform(pd.DataFrame(data=x[:, :self.addb.redis_len], 
                                                                      columns=self.addb.redis_knobs_columns))).cuda()
        dbms2 = torch.Tensor(self.addb.scaler_rocksdb.transform(pd.DataFrame(data=x[:, self.addb.redis_len:self.addb.redis_len+self.addb.rocksdb_len],
                                                                       columns=self.addb.rocksdb_knobs_columns))).cuda()
        dbms3 = torch.Tensor(self.addb.scaler_spark.transform(pd.DataFrame(data=x[:, -self.addb.spark_len:],
                                                                     columns=self.addb.spark_knobs_columns))).cuda()
        
        outputs = self.model(dbms1, dbms2, dbms3)
        outputs = np.round(self.addb.scaler_y.inverse_transform(outputs.cpu().detach().numpy()), 2)
        out["F"] = np.average(outputs, axis=-1)

class RocksDBMultiProblem(Problem):
    def __init__(self, addb, model):
        self.addb = addb
        self.model = model
        self.model.eval()
        n_var = sum([self.addb.redis_len, self.addb.rocksdb_len, self.addb.spark_len])
        n_obj = 1
        xl = addb.addb_lower_boundary
        xu = addb.addb_upper_boundary
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Torch(self.knobs.scaler_X.transform(x)).cuda()
        
        outputs = self.model(x)
        outputs = np.round(self.addb.scaler_y.inverse_transform(outputs.cpu().detach().numpy()), 2)
        out["F"] = np.average(outputs, axis=-1)
        
def genetic_algorithm(problem, pop_size, eliminate_duplicates=True):
    algorithm = GA(pop_size=pop_size, eliminate_duplicates=eliminate_duplicates)
    res = minimize(problem, algorithm, verbose=False)
    return res