# _*_ coding:utf-8 _*_

import tensorflow as tf
from app.util import RnnData
import pandas as pd
import time

# 参数设置
BATCH_SIZE = 100        # BATCH的大小，相当于一次处理50个image
TIME_STEP = 23          # 一个LSTM中，输入序列的长度，image有18行
INPUT_SIZE = 15         # x_i 的向量长度，image有28列
LR = 0.01               # 学习率
NUM_UNITS = 100         # 多少个LSTM单元
ITERATIONS= 10000        # 迭代次数
N_CLASSES= 6            # 输出大小，0-5十个数字的概率

# 定义 placeholders 以便接收x,y
# 维度是[BATCH_SIZE，TIME_STEP * INPUT_SIZE]
train_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])
# 输入的是二维数据，将其还原为三维，维度是[BATCH_SIZE, TIME_STEP, INPUT_SIZE]
image = tf.reshape(train_x, [-1, TIME_STEP, INPUT_SIZE])
train_y = tf.placeholder(tf.int32, [None, N_CLASSES])


# 定义RNN（LSTM）结构
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)

outputs,final_state = tf.nn.dynamic_rnn(
    cell=rnn_cell,              # 选择传入的cell
    inputs=image,               # 传入的数据
    initial_state=None,         # 初始状态
    dtype=tf.float32,           # 数据类型
    time_major=False)           # False: (batch, time step, input)
                                # True: (time step, batch, input)
                                # 这里根据image结构选择False

output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES)


loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)      #选择优化方法

y_pre = tf.arg_max(output, 1)

correct_prediction = tf.equal(tf.argmax(train_y, axis=1),tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))  #计算正确率

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # 初始化计算图中的变量

rnn_data = RnnData()
v_x, v_y = rnn_data.get_validation_data()
test_data , user_id = rnn_data.get_test_data()

get_train_data = rnn_data.get_train_batch(BATCH_SIZE)
for step in range(1, ITERATIONS+1):    # 开始训练
    train_data= next(get_train_data)
    tr_x, tr_y = train_data[0], train_data[1]
    _, loss_ = sess.run([train_op, loss], {train_x: tr_x, train_y: tr_y})
    if step % 500 == 0:
        acc_ =sess.run(accuracy, {train_x: v_x, train_y: v_y})
        print('train step {0}, loss: {1:.4f}, validation accuracy: {2:.2f}'.\
            format(step, loss_, acc_))

y_prediction = list(sess.run(y_pre, {train_x: test_data}))

dfMap = {"user_id": user_id, "prediction": y_prediction}
df = pd.DataFrame(dfMap)
df = df.loc[df["prediction"] >= 2]
predictions = df["user_id"].tolist()

pre_len = '_' + str(len(predictions))
currentTime = time.strftime("%Y%m%d_%H_%M", time.localtime())
print('start write...')
with open("../result/result" + str(currentTime) + pre_len + ".txt", "w") as file:
    for data in predictions:
        file.write(str(data) + '\n')

print('ok.')






