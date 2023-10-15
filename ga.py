# Genetic algorithm in MetaTune


class RocksDBSingleProblem(Problem):
    def __init__(self, knobs, model):
        self.knobs = knobs
        self.model = model
        self.model.eval()
        n_var = len(self.knobs.columns)
        n_obj = 1
        self.ex_len = self.knobs.default_trg_em.shape[-1] # length of external metrics
        xl = self.knobs.lower_boundary
        xu = self.knobs.upper_boundary
        
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(self.knobs.scaler_X.transform(x)).cuda()
        
        outputs = self.model(x, trg_len=self.ex_len, train=False)
        scaled_def = self.knobs.scaler_y.transform(self.knobs.default_trg_em)
        
        outputs = self.single_score_function(scaled_def, outputs.cpu().detach().numpy().squeeze())
        # outputs = self.single_score_function(self.knobs.default_trg_em, outputs.cpu().detach().numpy().squeeze())
        out["F"] = outputs
        
    def single_score_function(self, df, pr):
        # df = np.repeat(df, pr.shape[0], axis=0)
        df = np.repeat(np.array(df), pr.shape[0], axis=0)
        
        
        score = (df[:,0] - pr[:,0]) + (pr[:,1] - df[:,1]) + (df[:,2] - pr[:,2]) + (df[:,3] - pr[:,3])
        return np.round(-score, 6)