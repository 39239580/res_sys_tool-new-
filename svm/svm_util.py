from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,OneClassSVM,SVC,SVR,l1_min_c


class SVM(object):
    def __init__(self,task_type="linearsvc"):
        self.task_type = task_type
        assert self.task_type in {"linearsvc","linearsvr","nusvc","nusvr","oneclasssvm","svc","svr","l1_min_c"}

        if self.task_type == "linearsvc":   # 线性支持向量分类
            self.model = LinearSVC(penalty='l2',
                                   loss='squared_hinge',
                                   dual=True,
                                   tol=1e-4,
                                   C=1.0,
                                   multi_class='ovr',
                                   fit_intercept=True,
                                   intercept_scaling=1,
                                   class_weight=None,
                                   verbose=0,
                                   random_state=None,
                                   max_iter=1000)

        elif self.task_type =="linearsvr":  # 线性支持向量回归
            self.model = LinearSVR(epsilon=0.0,
                                   tol=1e-4, C=1.0,
                                   loss='epsilon_insensitive',
                                   fit_intercept=True,
                                   intercept_scaling=1.,
                                   dual=True,
                                   verbose=0,
                                   random_state=None,
                                   max_iter=1000)

        elif self.task_type =="nusvc":  # Nu 支持向量分类
            self.model = NuSVC(nu=0.5,
                               kernel='rbf',
                               degree=3,
                               gamma='scale',
                               coef0=0.0,
                               shrinking=True,
                               probability=False,
                               tol=1e-3,
                               cache_size=200,
                               class_weight=None,
                               verbose=False,
                               max_iter=-1,
                               decision_function_shape='ovr',
                               break_ties=False,
                               random_state=None)

        elif self.task_type =="nusvr":  # Nu支持向量回归
            self.model = NuSVR(nu=0.5,
                               C=1.0,
                               kernel='rbf',
                               degree=3,
                               gamma='scale',
                               coef0=0.0,
                               shrinking=True,
                               tol=1e-3,
                               cache_size=200,
                               verbose=False,
                               max_iter=-1)

        elif self.task_type =="oneclasssvm":  # 无监督异常值检测
            self.model = OneClassSVM(kernel='rbf',
                                     degree=3,
                                     gamma='scale',
                                     coef0=0.0,
                                     tol=1e-3,
                                     nu=0.5,
                                     shrinking=True,
                                     cache_size=200,
                                     verbose=False,
                                     max_iter=-1)

        elif self.task_type =="svc": # c支持向量分类
            self.model = SVC(C=1.0,
                             kernel='rbf',
                             degree=3,
                             gamma='scale',
                             coef0=0.0,
                             shrinking=True,
                             probability=False,
                             tol=1e-3,
                             cache_size=200,
                             class_weight=None,
                             verbose=False,
                             max_iter=-1,
                             decision_function_shape='ovr',
                             break_ties=False,
                             random_state=None)

        else: # Epsilion 支持向量回归
            self.model = SVR(kernel='rbf',
                             degree=3,
                             gamma='scale',
                             coef0=0.0,
                             tol=1e-3,
                             C=1.0,
                             epsilon=0.1,
                             shrinking=True,
                             cache_size=200,
                             verbose=False,
                             max_iter=-1)


    def decision_functions(self,x):
        self.model.decision_function(X=x)