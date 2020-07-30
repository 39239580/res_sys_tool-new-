import tensorflow as tf
from yellowfin import YFOptimizer
import os
from tensorflow.examples.tutorials.mnist import input_data


class LR(object):
    def __init__(self, input_node=784, output_node=10, learning_rate=0.8, training_epochs=300,
                 batch_size=32, save_step=10, dispaly_step=1,
                 optimizer_type="sgd", ifregularizer=True, regularaztion_rate=0.0001, loss_type="cross_entropy",
                 model_save_path="./LR_model/", model_name="model.ckpt"):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.save_step = save_step
        self.display_step = dispaly_step
        self.output_node = output_node
        self.optimizer_type = optimizer_type
        self.ifregularizer = ifregularizer
        self.regularaztion_rate = regularaztion_rate
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regularaztion_rate)
        self.loss_type = loss_type
        self.input_node = input_node
        self.model_save_path = model_save_path
        self.model_name = model_name
        self._init_graph()

    def _net(self, input_tenor, output_style="prob"):
        with tf.compat.v1.variable_scope("net", reuse=tf.compat.v1.AUTO_REUSE):
            weighted = tf.compat.v1.get_variable(name="weight",
                                                 shape=[self.input_node, self.output_node],
                                                 initializer=tf.compat.v1.random_normal_initializer(
                                                     mean=1.0,
                                                     stddev=0.01,
                                                     dtype=tf.float32
                                                 ))

            if self.ifregularizer:  # 使用正则化处理
                tf.compat.v1.add_to_collection("losses", self.regularizer(weighted))

            bias = tf.compat.v1.get_variable(name="baise",
                                             shape=[self.output_node],
                                             initializer=tf.compat.v1.random_uniform_initializer(
                                                 minval=0.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32
                                             ))
            layer = tf.add(tf.matmul(input_tenor, weighted), bias, name="out")

            logits = tf.compat.v1.nn.softmax(layer, name="logits")
            return layer, logits

    def _optimizer(self, global_step):  # 选择优化器
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

    def _loss(self, logits, label):  # 损失函数
        if self.loss_type == "logloss":
            loss = tf.compat.v1.losses.log_loss(labels=label, predictions=logits)
            loss = tf.compat.v1.reduce_mean(loss)  # 误差求均值求均值

        elif self.loss_type == "mse":
            loss = tf.compat.v1.nn.l2_loss(tf.subtract(label, logits))
            loss = tf.compat.v1.reduce_mean(loss)  # 误差求均值求均值

        elif self.loss_type == "square_loss":
            loss = tf.compat.v1.sqrt(tf.reduce_mean(
                tf.compat.v1.squared_difference(tf.reshape(logits, [-1]),
                                                tf.reshape(label, [-1]))
            ))

        else:  # 交叉熵
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.math.argmax(label, 1))
            loss = tf.compat.v1.reduce_mean(loss)  # 误差求均值求均值
        return loss

    def _init_graph(self):
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_node])
        self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, self.output_node])
        self.y, y_porb = self._net(self.x, output_style="pred")
        self.global_step = tf.compat.v1.Variable(0, trainable=False, name="step")  # 初始话变量为0， 不可训练
        losses = self._loss(self.y, self.y_)
        self.loss = losses + tf.add_n(tf.compat.v1.get_collection("losses"))
        tf.compat.v1.add_to_collection("loss", self.loss)
        train_step = self._optimizer(global_step=self.global_step)
        train_ops = [train_step]
        with tf.control_dependencies(train_ops):
            self.train_op = tf.no_op(name="train")

    def train(self, data):
        # 初始化
        saver = tf.compat.v1.train.Saver(max_to_keep=5)  # 默认为5个模型。更改为保存最多100个模型
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()
            # 等价于 init_opt = tf.global_variables_initialzer()
            # sess.run(init_opt)
            # 加入断点继续训练
            ckpt = tf.train.get_checkpoint_state(self.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")
            else:
                print("模型不存在， 需要重新训练")
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(data.train.num_examples / self.batch_size)
                # Loop over all batches
                for j in range(total_batch):
                    xs, ys = data.train.next_batch(self.batch_size)  # 训练集数据
                    _, loss_value, step = sess.run([self.train_op, self.loss, self.global_step],
                                                   feed_dict={self.x: xs, self.y_: ys})
                    # Compute average loss
                    avg_cost += loss_value / total_batch
                # 每一千轮保存一次模型
                if epoch % self.save_step == 0:
                    saver.save(sess, os.path.join(self.model_save_path, self.model_name),
                               global_step=self.global_step)

                if (epoch + 1) % self.display_step == 0:
                    print("After %d traning step(s),loss on training batch is %g" % (epoch + 1, avg_cost))
            self._evpoch(data)

    def _evpoch(self, data):
        print("Optimization Finished!")
        # Test model
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # cast(x, dtype, name=None) 将x的数据格式转化成dtype.
        print("Accuracy:", accuracy.eval({self.x: data.test.images[:3000], self.y_: data.test.labels[:3000]}))


def fn():
    mnist = input_data.read_data_sets("./MNIST/MNIST_data", one_hot=True)  # 下载的数据集
    tf.compat.v1.reset_default_graph()  # 重置计算图
    lr = LR()
    lr.train(mnist)


if __name__ == "__main__":
    fn()
