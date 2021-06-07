import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from resnet import resnet34

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


# 数据预处理，仅仅是类型的转换。    [-1~1]
def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 数据集的加载
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
y_test = tf.squeeze(y_test)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(512)

# 我们来测试一下sample的形状。
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))  # 值范围为[0,1]


def main():
    # 输入：[b, 32, 32, 3]
    model = resnet34()
    model.build(input_shape=(None, 512, 512, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)


    for epoch in range(500):

        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model(x)
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth=100)
                # compute loss   结果维度[b]
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            # 梯度求解
            grads = tape.gradient(loss, model.trainable_variables)
            # 梯度更新
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print(epoch, step, 'loss:', float(loss))

        # 做测试
        total_num = 0
        total_correct = 0
        for x, y in test_db:

            logits = model(x)
            # 预测可能性。
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)  # 还记得吗pred类型为int64,需要转换一下。
            pred = tf.cast(pred, dtype=tf.int32)

            # 拿到预测值pred和真实值比较。
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)  # 转换为numpy数据

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()