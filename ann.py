import tensorflow as tf


class ArtificialNeuralNetwork:

    def __init__(self, epochs=10, learning_rate=0.001):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output = self.model()
        self.optimizer, self.cost = self.backprop(self.output, self.y)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def model(self):
        self.X = tf.placeholder(tf.float32, shape=(None, 400), name="features")
        self.y = tf.placeholder(tf.float32, shape=(None, 2), name="labels")

        x1 = tf.layers.dense(self.X, units=200)
        a1 = tf.nn.tanh(x1)

        x2 = tf.layers.dense(a1, units=100)
        a2 = tf.nn.tanh(x2)

        x3 = tf.layers.dense(a2, units=2)

        return x3

    def backprop(self, logits, labels):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cost"))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        return optimizer, cost

    def fit(self, features, labels):
        for e in range(self.epochs):
            _, c = self.sess.run([self.optimizer, self.cost], feed_dict={
                self.X: features,
                self.y: labels
            })
            print(c)
        return self

    def predict(self, features):
        outputs = self.sess.run(self.output, feed_dict={
            self.X: features
        })
        return outputs


if __name__ == "__main__":
    import numpy as np
    X = np.random.randn(5, 400)
    y = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0]
    ])

    print(X.shape)
    print(y.shape)

    ann = ArtificialNeuralNetwork(epochs=50)
    ann.fit(X, y)
    result = ann.predict(X)
    print(result)
    print(result.shape)