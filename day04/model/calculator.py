import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Calculator:
    def create_adder(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2':2.0}
        r = tf.add(w1, w2, name='op_add')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print('덧셈결과: {}'.format(result))
        saver.save(sess, './calc_add_model/model', global_step=1000)

    def create_subber(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.subtract(w1, w2, name='op_sub')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print('뺄셈결과: {}'.format(result))
        saver.save(sess, './calc_sub_model/model', global_step=1000)

    def create_multifier(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.multiply(w1, w2, name='op_mul')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print('곱셈결과: {}'.format(result))
        saver.save(sess, './calc_mul_model/model', global_step=1000)

    def create_divder(self, num1, num2, opcode):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.divide(w1, w2, name='op_div')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print('나눗셈결과: {}'.format(result))
        saver.save(sess, './calc_div_model/model', global_step=1000)


def service(num1, num2, opcode):
    tf.reset_default_graph()  # 초기화
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(f'model/calc_{opcode}_model/model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint(f'model/calc_{opcode}_model'))
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name('w1:0')
        w2 = graph.get_tensor_by_name('w2:0')
        feed_dict = {w1: float(num1), w2: float(num2)}
        for key in feed_dict.keys():
            print(key, ':', feed_dict[key])
        op_to_restore = graph.get_tensor_by_name(f'op_{opcode}:0')
        result = sess.run(op_to_restore, feed_dict)
        print('텐서가 계산한 결과:{}'.format(result))
    return result

if __name__ == '__main__':
    c = Calculator()
    # c.create_adder()
    # c.create_subber()
    # c.create_multifier()
    c.create_divder()
