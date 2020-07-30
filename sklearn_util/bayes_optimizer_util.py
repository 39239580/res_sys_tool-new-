from bayes_opt import BayesianOptimization


class BayesianSearch(object):
    def __init__(self, f, pbounds, random_state=None, vernose=2, bounds_transformer=None):
        """
        Parameters
        ----------
        f : TYPE
            DESCRIPTION.    评估器
        pbounds : TYPE   {}
            DESCRIPTION. 参数取值范围
        random_state : TYPE, optional
            DESCRIPTION. The default is None.
        vernose : TYPE, optional
            DESCRIPTION. The default is 2.
        bounds_transformer : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.optimizer = BayesianOptimization(f, pbounds, random_state=random_state,
                                              verbose=vernose,
                                              bounds_transformer=bounds_transformer)
    
    #  进行参数空间搜索
    def maximize(self, init_points=5, n_iter=25, acq='ucb',
                 kappa=2.576, kappa_decay=1, kappa_decay_delay=0,
                 xi=0.0):
        self.optimizer.maximize(   init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 )
    
    # 输出最佳参数
    def get_max(self):
        return self.optimizer.max
