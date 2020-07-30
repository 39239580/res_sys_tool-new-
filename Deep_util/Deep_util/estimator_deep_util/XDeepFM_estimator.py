import tensorflow as tf
from tensorflow import estimator


# 相关带代码参考 https://github.com/wangruichens/recsys/blob/master/xdeepfm/xdeepfm.py
class XDeepFMEstimator(object):
    def __init__(self, config, model_params, model_dir):
        self.config = config
        self.model_params = model_params
        self.model = estimator.Estimator(model_fn=self.my_model,
                                         model_dir=model_dir,
                                         params=self.model_params,
                                         config=self.config
                                         )

    @staticmethod
    def my_model(features, labels, mode, params):
        layers = list(map(int, params["deep_layers"].split(",")))
        cross_layers = list(map(int, params["cross"]))

        # 输入层
        liner_net = tf.compat.v1.feature_column.input_layer(features, params["linear_feature_columns"])
        embedding_net = tf.compat.v1.feature_column.input_layer(features, params["embedding_feature_columns"])

        with tf.compat.v1.name_scope(name="linear_net"):
            liner_y = tf.compat.v1.layers.dense(liner_net, 1, activation=tf.nn.relu)
            # 加BN 层
            liner_y = tf.compat.v1.layers.batch_normalization(inputs=liner_y,
                                                              training=(mode == estimator.ModeKeys.TRAIN))
            liner_y = tf.compat.v1.layers.dropout(liner_y, rate=params["dropout"],
                                                  training=(mode == estimator.ModeKeys.TRAIN))

        with tf.compat.v1.name_scope(name="cin_net"):
            field_nums = []  # 字段个数
            hidden_nn_layes = []
            final_len = 0
            final_result = []
            cin_net = tf.compat.v1.reshape(embedding_net, shape=[-1, len(params["embedding_feature_columns"]),
                                                                 params["embedding_size"]])
            field_nums.append(len(params["embedding_feature_columns"]))
            hidden_nn_layes.append(cin_net)

            split_tensor0 = tf.compat.v1.split(hidden_nn_layes[0], params["embedding_size"]*[1], 2)
            for idx, layer_size in enumerate(cross_layers):
                split_tensor = tf.compat.v1.split(hidden_nn_layes[-1], params["embedding_size"]*[1], 2)
                dot_result_m = tf.compat.v1.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.compat.v1.reshape(dot_result_m, shap=[params["embedding_size"], -1,
                                                                        field_nums[0]*field_nums[-1]])
                dot_result = tf.compat.v1.transpose(dot_result_o, perm=[1, 0, 2])
                filters = tf.compat.v1.get_variable(name="f_"+str(idx),
                                                    shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                                    dtype=tf.float32)
                curr_out = tf.compat.v1.nn.conv1d(dot_result, filters=filters, stride=1, padding="VALID")

                # 添加偏置

                b = tf.compat.v1.get_variable(name="f_b" + str(idx),
                                              shape=[layer_size],
                                              dtype= tf.float32,
                                              initializer=tf.compat.v1.zeros_initializer())
                curr_out = tf.compat.v1.nn.bias_add(curr_out, b)

                # 激活函数
                curr_out = tf.compat.v1.nn.relu(curr_out)
                curr_out = tf.compat.v1.transpose(curr_out, perm=[0, 2, 1])

                # 直接相连
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                field_nums.append(int(layer_size))

                final_result.append(direct_connect)
                hidden_nn_layes.append(next_hidden)
            result = tf.compat.v1.concat(final_result, axis=1)
            result = tf.compat.v1.reduce_sum(result, -1)

            cin_y = tf.compat.v1.layers.dense(result, 1, activation=tf.compat.v1.nn.relu)

        with tf.compat.v1.name_scope(name="dnn_net"):
            embedding_net = tf.compat.v1.feature_column.input_layer(features=features,
                                                                    feature_columns=params["embedding_feature_columns"])
            dnn_net = tf.compat.v1.reshape(embedding_net,
                                           shape=[-1,
                                                  len(params["embedding_feature_columns"]*params["embedding_size"])])
            for i in layers:
                dnn_net = tf.compat.v1.layers.dense(dnn_net, i, activation=tf.compat.v1.nn.relu)
                dnn_net = tf.compat.v1.layers.batch_normalization(dnn_net, training=(mode == estimator.ModeKeys.TRAIN))
                dnn_net = tf.compat.v1.layers.dropout(dnn_net,rate=params["dropout"],
                                                      training=(mode == estimator.ModeKeys.TRAIN))
            dnn_y = tf.compat.v1.layers.dense(dnn_net, 1, activation=tf.compat.v1.nn.relu)

        logits = tf.compat.v1.concat([liner_y, cin_y, dnn_y], axis=-1)
        logits = tf.compat.v1.layers.dense(logits, untis=1)   # 最后为一个节点的数据
        pred = tf.compat.v1.sigmoid(logits)

        predictions = {"prob": pred}
        export_outputs = {
            tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                estimator.export.PredictOutput(predictions)
        }

        # 预测的值
        if mode == estimator.ModeKeys.PREDICT:
            return estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                                          labels=tf.cast(layers,
                                                                                                         tf.float32)))

        eval_metric_ops = {
            "AUC": tf.compat.v1.metrics.auc(labels=labels, predictions=pred),
            "Accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=pred)
        }

        if mode == estimator.ModeKeys.EVAL:
            return estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )

        # 训练中要用到的优化器
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss,global_step=tf.compat.v1.train.global_step())
        if mode == estimator.ModeKeys.TRAIN:
            return estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op
            )

    def train(self, filename, batch_size, input_fn, num_epoches, need_shuffle=True,
              hooks=None,steps=None, max_steps=None,saving_listeners=None):
        self.model.train(input_fn=lambda: input_fn(
            filename=filename,
            batch_size=batch_size,
            num_epoches=num_epoches,
            need_shuffle=need_shuffle),
                         hooks=hooks,
                         steps=steps,
                         max_steps=max_steps,
                         saving_listeners=saving_listeners)

    def predict(self, filename, batch_size, input_fn, num_epoches=1, need_shuffle=True,
                predict_keys=None, hooks=None, checkpoint_path=None,
                yield_single_examples=True):
        predict = self.model.predict(input_fn=lambda: input_fn(
            filename=filename,
            batch_size=batch_size,
            num_epoches=num_epoches,
            need_shuffle=need_shuffle),
                                     predict_keys=predict_keys,
                                     hooks=hooks,
                                     checkpoint_path=checkpoint_path,
                                     yield_single_examples=yield_single_examples
                                     )
        return predict

    def evaluate(self, filename, batch_size, input_fn, num_epoches=1, need_shuffle=False,
                 steps=None, hooks=None, checkponit_path=None):
        self.model.evaluate(input_fn=lambda: input_fn(
            filename=filename,
            batch_size=batch_size,
            num_epoches=num_epoches,
            need_shuffle=need_shuffle),
                            steps=steps,
                            hooks=hooks,
                            checkpoint_path=checkponit_path)
