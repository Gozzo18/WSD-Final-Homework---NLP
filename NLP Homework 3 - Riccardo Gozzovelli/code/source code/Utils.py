import tensorflow as tf

def batch_generator(X, Y, sequence_length, batch_size):
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end], Y[start:end], sequence_length[start:end]

def alternative_batch_generator(X, Y, domain_Y, lexname_Y, sequence_length, batch_size):
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end], Y[start:end], domain_Y[start:end], lexname_Y[start:end], sequence_length[start:end]

def add_summary(writer, name, value, global_step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, global_step=global_step)
