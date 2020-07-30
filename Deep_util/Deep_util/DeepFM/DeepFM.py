import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, roc_curve  # 性能评估API
from time import time
from yellowfin import YFOptimizer
# from tensorflow.compat.v1.layers import batch_normalization as batch_norm
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
# from tensorflow.contrib.layers.python.layers import l2_regularizer, l1_regularizer, l1_l2_regularizer  # 1.x版本
from tensorflow.keras.regularizers import l2, l1_l2,l1
# from tensorflow.keras.regularizers import l1,l2, l1_l2
# 1.14 可用， 在2.0  被内置到 keras 中去了


class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_size,   # 记作M, 特征字典的大小
                 field_size,    # 记作F。 特征字段的大小 filed的大小
                 embedding_size=8,   # 记为K。 特征嵌入的尺寸
                 deep_layers=[32, 32],
                 dropout_fm=[1.0, 1.0],
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activateion="relu",
                 epoch=10,  # 进行10轮训练
                 batch_size=64,
                 learning_rate=0.001,
                 optimizer_type="adam",
                 batch_norms=None,
                 batch_norm_decay=0.995,
                 verbose=False,
                 random_seed=2018,
                 module_type="DeepFM",
                 loss_type="logloss",
                 eval_metric="auc",
                 regularizer_type ="l2",
                 l2_reg=0.0,  # l2 lambda  系数
                 l1_reg=0.0,  # l1 lambda 系数
                 greater_is_better=True):
        assert module_type in {"DeepFM", "FM", "Deep", None}
        assert loss_type in {"logloss", "mse", "crossentropy", None}
        assert deep_layers_activateion in {"relu", "sigmoid", "tanh", "leakrelu",None}
        assert optimizer_type in {"sgd", "adam", "yellowfin", None}  # 等 相关优化器
        assert isinstance(dropout_deep, list) or isinstance(dropout_deep, int)
        assert isinstance(dropout_fm, list) or isinstance(dropout_fm, int)
        assert isinstance(deep_layers, list)
        # 数据相关参数
        self.feature_size = feature_size   # 总的特征维

        self.field_size = field_size   # 字段的大小
        self.embedding_size = embedding_size  # embedding层的大小
        # 模型相关参数
        self.dropout_deep = dropout_deep
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.deep_layers_activateion = deep_layers_activateion
        self.module_type = module_type
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.regularizer_type = regularizer_type
        self.epoch = epoch  # 代数
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norms = batch_norms
        self.batch_norm_decay = batch_norm_decay
        # 辅助参数
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

    def _init_graph(self):
        self.graph = tf.compat.v1.Graph()  # 创建新的计算图
        with self.graph.as_default():  # 作为默认的计算图
            tf.compat.v1.set_random_seed(self.random_seed)
            # 占位符
            self.feat_index = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="feat_index")  # None*F
            self.feat_value = tf.compat.v1.placeholder(tf.float32, shape=[None, None], name="feat_value")  # None*F
            self.label = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="label")   # None*1
            self.dropout_keep_fm = tf.compat.v1.placeholder(tf.float32, shape=[None], name="fm_dropout")
            self.dropout_keep_deep = tf.compat.v1.placeholder(tf.float32, shape=[None], name="deep_dropout")
            self.train_phase = tf.compat.v1.placeholder(tf.bool, name="train_phase")  # 用于判断先练阶段与推理阶段
            self.weights = self._initialize_weights()  # 初始化权重

            # ------------------------model---------------------------
            self.embeddings = tf.compat.v1.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)
            # None*F*K

            feat_value = tf.compat.v1.reshape(self.feat_value, shape=[-1, self.field_size, 1])

            self.embeddings = tf.compat.v1.multiply(self.embeddings, feat_value)   #

            # --------------------------first order term---------------------------------
            self.y_first_order = tf.compat.v1.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index)
            # None*F*1
            self.y_first_order = tf.compat.v1.reduce_mean(tf.compat.v1.multiply(self.y_first_order, feat_value), axis=2)
            # None*F
            self.y_first_order = tf.compat.v1.nn.dropout(self.y_first_order, rate=1-self.dropout_fm[0])
            # None*F

            # ---------------------------second order term-------------------------------
            # ---------------------------sum_square part---------------------------------
            self.summed_features_emb = tf.compat.v1.reduce_sum(self.embeddings, axis=1)
            self.summed_features_emb_square = tf.compat.v1.square(self.summed_features_emb)  # None*K

            # ---------------------------square_sum part---------------------------------
            self.squared_features_emb = tf.compat.v1.square(self.embeddings)
            self.squared_sum_features_emb = tf.compat.v1.reduce_sum(self.squared_features_emb, axis=1)  # None*K

            # second order
            self.y_second_order = 0.5*tf.compat.v1.subtract(self.summed_features_emb_square,
                                                            self.squared_sum_features_emb)
            self.y_second_order = tf.compat.v1.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            # ----------------------------DeepFM component-------------------------------
            self.y_deep = tf.compat.v1.reshape(self.embeddings, shape=[-1, self.field_size*self.embedding_size])
            # None*(F*K)
            self.y_deep = tf.compat.v1.nn. dropout(self.y_deep, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_layers)):  # 遍历每个DNN层
                self.y_deep = tf.compat.v1.add(tf.compat.v1.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                               self.weights["bias_"+str(i)])  # None*layer[i]*1
                if self.batch_norms:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                        scope_bn="bn_"+str(i))

                self.y_deep = self._activate(logit=self.y_deep, activation=self.deep_layers_activateion)
                self.y_deep = tf.compat.v1.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i])  #

            # -------------------------------use DeepFM---------------------------------
            if self.module_type == "DeepFM":
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.module_type == "FM":
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.module_type == "Deep":
                concat_input = self.y_deep
            else:
                ValueError("Error use module_type")
            self.out = tf.compat.v1.add(tf.compat.v1.matmul(concat_input, self.weights["concat_weight"]),
                                        self.weights["concat_bias"])

            # ---------------------------------loss-------------------------------------
            if self.loss_type == "logloss":
                self.out = tf.compat.v1.nn.sigmoid(self.out)
                self.loss = tf.compat.v1.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.compat.v1.nn.l2_loss(tf.subtract(self.label, self.out))

            # ----------------------------------------1.x版本--------------------------------------
            # 在公式中加入l2正则化
            if self.regularizer_type == "l1":
                self.loss += l1(self.l1_reg)(self.weights["concat_weight"])
                if self.module_type == "DeepFM":
                    for i in range(len(self.deep_layers)):
                        self.loss += l1(self.l1_reg)(self.weights["layer_"+str(i)])
            elif self.regularizer_type == "l1_l2":
                self.loss += l1_l2(self.l1_reg, self.l2_reg)(self.weights["concat_weight"])
                if self.module_type == "DeepFM":
                    for i in range(len(self.deep_layers)):
                        self.loss += l1_l2(self.l1_reg, self.l2_reg)(self.weights["concat_weight"])
            elif self.regularizer_type == "l2":
                self.loss += l2(self.l1_reg)(self.weights["concat_weight"])
                if self.module_type == "DeepFM":
                    for i in range(len(self.deep_layers)):
                        self.loss += l2(self.l1_reg)(self.weights["layer_" + str(i)])
            else:
                pass

            # # ------------------------------------2.x版本-------------------------------------------------
            # if self.regularizer_type == "l1":
            #     self.loss += l1(self.l1_reg)(self.weights["concat_weight"])
            #     if self.module_type == "DeepFM":
            #         for i in range(len(self.deep_layers)):
            #             self.loss += l1(self.l1_reg)(self.weights["layer_"+str(i)])
            # elif self.regularizer_type == "l1_l2":
            #     self.loss += l1_l2(self.l1_reg, self.l2_reg)(self.weights["concat_weight"])
            #     if self.module_type == "DeepFM":
            #         for i in range(len(self.deep_layers)):
            #             self.loss += l1_l2(self.l1_reg, self.l2_reg)(self.weights["concat_weight"])
            # elif self.regularizer_type == "l2":
            #     self.loss += l2(self.l1_reg)(self.weights["concat_weight"])
            #     if self.module_type == "DeepFM":
            #         for i in range(len(self.deep_layers)):
            #             self.loss += l2(self.l1_reg)(self.weights["layer_" + str(i)])
            # else:
            #     pass

            self.optimizer = self._optimizer()  # 使用对应的优化器
            # __init__
            self.saver = tf.compat.v1.train.Saver()
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
        # num of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        if self.verbose > 0:
            print("#params:%d" % total_parameters)

    @staticmethod
    def _init_session():  # 初始化会话
        config = tf.compat.v1.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)

    def _initialize_weights(self):
        weights = dict()
        # --------------------------embedding part--------------------------
        weights["feature_embeddings"] = tf.compat.v1.Variable(
            tf.compat.v1.random_normal([self.feature_size, self.embedding_size], mean=0.0, stddev=0.01),  # 正态分布的权重
            dtype=tf.float32, name="feature_embeddings")  # feature_size*embdeding_size

        weights["feature_bias"] = tf.compat.v1.Variable(
            tf.compat.v1.random_uniform([self.feature_size, 1], minval=0.0, maxval=1.0), dtype=tf.float32,
            name="feature_bias")  # feature_size*1 的均匀分布的

        # --------------------------deep part-------------------------------
        num_layer = len(self.deep_layers)  # Deep 的层数
        input_size = self.field_size*self.embedding_size  # filed字段数 *嵌入之后的长度
        for i in range(num_layer):
            if i == 0:
                glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
                weights["deep_layer_" + str(i)] = tf.compat.v1.Variable(
                    tf.compat.v1.random_normal([input_size, self.deep_layers[0]], mean=0.0, stddev=glorot),
                    dtype=tf.float32, name="deep_layer_" + str(i))  #
                weights["deep_layer_" + str(i) + "_baise"] = tf.compat.v1.Variable(
                    tf.compat.v1.random_normal([1, self.deep_layers[0]], mean=0.0, stddev=glorot),
                    dtype=tf.float32, name="deep_layer_" + str(i) + "_baise")
            else:
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                weights["deep_layer_" + str(i)] = tf.compat.v1.Variable(
                    tf.compat.v1.random_normal([self.deep_layers[i-1], self.deep_layers[i]], mean=0.0, stddev=glorot),
                    dtype=tf.float32, name="deep_layer_" + str(i))  #
                weights["deep_layer_" + str(i) + "_baise"] = tf.compat.v1.Variable(
                    tf.compat.v1.random_normal([1, self.deep_layers[i]], mean=0.0, stddev=glorot),
                    dtype=tf.float32, name="deep_layer_" + str(i) + "_baise")

        # -----------------------final_concat part-----------------------------
        if self.module_type == "DeepFM":
            input_size = self.field_size + self.embedding_size+self.deep_layers[-1]

        elif self.module_type == "FM":
            input_size = self.field_size + self.embedding_size

        elif self.module_type == "Deep":
            input_size = self.field_size + self.deep_layers[-1]

        else:
            ValueError("please Enter correct moduletype!")

        glorot = np.sqrt(2.0/(input_size+1))
        weights["concat_weight"] = tf.compat.v1.Variable(
            tf.compat.v1.random_normal([input_size, 1], mean=0.0, stddev=glorot),
            dtype=tf.float32, name="concat_weight")
        weights["concat_bias"] = tf.compat.v1.Variable(
            tf.compat.v1.constant(0.01), dtype=tf.float32
        )
        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)

        z = tf.compat.v1.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    @staticmethod
    def get_batch(xi, xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return xi[start:end], xv[start:end], [[y_] for y_ in y[start:end]]

        # shuffle three lists simutaneously

    @staticmethod
    def shuffle_in_unison_scary(a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, xi, xv, y):
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, xi_train, xv_train, y_train,
            xi_valid=None, xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24,
                         for numerical features)
        :param y_train: label of each sample in the training set
        :param xi_valid: list of list of feature indices of each sample in the validation set
        :param xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = xv_valid is not None   # 有验证集合的情况
        for epoch in range(self.epoch):    # 每一轮的操作
            t1 = time()
            self.shuffle_in_unison_scary(xi_train, xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                xi_batch, xv_batch, y_batch = self.get_batch(xi_train, xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(xi_batch, xv_batch, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(xi_train, xv_train, y_train)
            self.train_result.append(train_result)
            valid_result = None
            if has_valid:
                valid_result = self.evaluate(xi_valid, xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score,
        if has_valid and refit:   # 已经达到最佳状态
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            xi_train = xi_train + xi_valid
            xv_train = xv_train + xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(xi_train, xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(xi_train, xv_train, y_train,
                                                                 self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(xi_train, xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and valid_result[-2] < valid_result[-3] and valid_result[-3]\
                        < valid_result[-4] and valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        if self.eval_metric == "auc":
            return  roc_auc_score(y, y_pred)
        elif self.eval_metric == "roc":
            fpr, tpr, threshold=roc_curve(y_true=y, y_score=y_pred)
            return fpr, tpr, threshold
        else:
            raise ValueError("指标有误， 请输出 auc , roc 中的一种")

    def _optimizer(self, global_step=None):  # 选择优化器
        if self.optimizer_type == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                         beta1=0.9, beta2=0.999, epsilon=1e-8
                                                         ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "adadelta":
            optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learning_rate
                                                             ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "adagrad":
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                            initial_accumulator_value=1e-8
                                                            ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "gd" or self.optimizer_type == "sgd":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate
                                                                    ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "ftrl":
            optimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate=self.learning_rate
                                                         ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "padagrad":
            optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate
                                                                    ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "pgd":
            optimizer = tf.compat.v1.train.ProximalGradientDescentOptimizer(learning_rate=self.learning_rate
                                                                            ).minimize(self.loss,
                                                                                       global_step=global_step)

        elif self.optimizer_type == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                                            ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "momentum":
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                             momentum=0.2
                                                             ).minimize(self.loss, global_step=global_step)

        elif self.optimizer_type == "yellowfin":
            optimizer = YFOptimizer(learning_rate=self.learning_rate,
                                    momentum=0.0).minimize(self.loss, global_step=global_step)

        else:
            raise ValueError("this optimizer is undefined{0}".format(self.optimizer_type))

        return optimizer

    @staticmethod  # 静态方法
    def _activate(logit, activation):  # 激活函数
        if activation == "sigmoid":
            return tf.compat.v1.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.compat.v1.nn.softmax(logit)
        elif activation == "relu":
            return tf.compat.v1.nn.relu(logit)
        elif activation == "tanh":
            return tf.compat.v1.nn.tanh(logit)
        elif activation == "elu":
            return tf.compat.v1.nn.elu(logit)
        elif activation == "identity":
            return tf.compat.v1.identity(logit)
        elif activation == "leaky_relu":
            return tf.compat.v1.nn.leaky_relu(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))
