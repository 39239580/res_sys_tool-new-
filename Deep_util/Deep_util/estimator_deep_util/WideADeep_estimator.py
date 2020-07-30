from Deep_util.common_base_estimator_tf_util.common_api_estimator import CommonEstimator
import tensorflow as tf


class WideAndDeep(CommonEstimator):
    def __init__(self, model_type, params, task_type=None, device=None, wide_columns=None, deep_columns=None):
        """
        :param model_type: 模型类型
        :param params: 相关的配置参数
        :param task_type: 任务类型　　分类还是回归
        :param device: 设备
        :param wide_columns:　wide 类型的列
        :param deep_columns:  deep　类型的列
        """
        assert task_type in {"class","classifier","CLASS","cla", "regression","ReGression","Regre","regre", None}
        assert model_type in {"wide","deep","wide_deep", None}
        assert device in {"cpu","CPU","GPU","gpu", None}
        assert (not wide_columns) or (not deep_columns)
        if not model_type:
            model_type = "wide_deep"
        if not task_type:
            task_type ="class"
        if not device:   # 若使用默认，则为gpu
            run_config = tf.compat.v1.estimator.RunConfig().replace(
                session_config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))
            params["config"] = run_config
        if model_type == "wide_deep":
            params["linear_feature_columns"] = wide_columns
            params["dnn_feature_columns"] = deep_columns
        elif model_type == "deep":
            params["feature_columns"] = deep_columns
        elif model_type == "wide":
            params["feature_columns"] = wide_columns
        else:
            raise  ValueError("check model_type please!")
        super(WideAndDeep, self).__init__(model_type=model_type, task_type=task_type, params=params)

    def fit(self, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):
        super(WideAndDeep, self).fit(input_fn=input_fn, steps=steps, max_steps=max_steps,
                                     saving_listeners=saving_listeners)

    def predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        return super(WideAndDeep, self).predict(input_fn=input_fn,predict_keys=predict_keys, hooks=hooks,
                                                checkpoint_path=checkpoint_path,
                                                yield_single_examples=yield_single_examples)

    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        super(WideAndDeep, self).evaluate(input_fn=input_fn, steps=steps, hooks=None, checkpoint_path=None, name=None)


