import tensorflow as tf

def simpleBiLSTM(batch_size, embedding_size, hidden_size, output_vocabulary_length): 
    """
	Bidirectional LSTM model. Inputs must already be embeddings.
    
    """

    inputs = tf.placeholder(tf.float32, shape=[None, None, embedding_size], name="Inputs")
    print("INPUTS TENSOR SIZE ", inputs)

    labels = tf.placeholder(tf.int64, shape=[None, None], name='Labels')
    print("LABEL TENSOR SIZE ", labels)
        
    input_prob = tf.placeholder(tf.float32, shape=[], name='Input_Probability')
    print("INPUT PROB. TENSOR ", input_prob)

    output_prob = tf.placeholder(tf.float32, shape=[], name='Output_Probability')
    print("OUTPUT PROB. TENSOR ", output_prob)

    state_prob = tf.placeholder(tf.float32, shape=[], name='State_Probability')
    print("STATE PROB. TENSOR ", state_prob)

    sequence_length = tf.placeholder(tf.int32, shape=[None], name="Sequence_Length")
    print("SEQUENCE_LENGTH TENSOR ", sequence_length)
    
    with tf.variable_scope("rnn_1"):
        rnn_fw_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell_1,
                                            input_keep_prob=input_prob,
                                            output_keep_prob=output_prob,
                                            state_keep_prob=state_prob,
                                            variational_recurrent=True,
                                            input_size=inputs.shape[-1],
                                            dtype=tf.float32)
        rnn_bw_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell_1,
                                            input_keep_prob=input_prob,
                                            output_keep_prob=output_prob,
                                            state_keep_prob=state_prob,
                                            variational_recurrent=True,
                                            input_size=inputs.shape[-1],
                                            dtype=tf.float32)
        
        (forward_output_1, backward_output_1), states_1 = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell_1, rnn_bw_cell_1, inputs,  sequence_length=sequence_length, dtype='float32')
        outputs_1 = tf.concat([forward_output_1, backward_output_1], axis=2)

    with tf.variable_scope("rnn_2"):
        rnn_fw_cell_2 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_fw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell_2,
                                            input_keep_prob=input_prob,
                                            output_keep_prob=output_prob,
                                            state_keep_prob=state_prob,
                                            variational_recurrent=True,
                                            input_size=outputs_1.shape[-1],
                                            dtype=tf.float32)
        rnn_bw_cell_2 = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_bw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell_2,
                                            input_keep_prob=input_prob,
                                            output_keep_prob=output_prob,
                                            state_keep_prob=state_prob,
                                            variational_recurrent=True,
                                            input_size=outputs_1.shape[-1],
                                            dtype=tf.float32)
        
        (forward_output_2, backward_output_2), states_2 = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell_2, rnn_bw_cell_2, outputs_1, sequence_length=sequence_length, dtype='float32')
        outputs_2 = tf.concat([forward_output_2, backward_output_2], axis=2)

    with tf.variable_scope("dense"):
        logits = tf.layers.dense(outputs_2, output_vocabulary_length, activation=None, name='Logits')
        softmax_logits = tf.nn.softmax(logits, name='Softmax_Logits')
        prediction = tf.argmax(softmax_logits, axis=-1, name='ArgMax_Predictions')    

    with tf.variable_scope("loss"):
        mask = tf.greater(tf.count_nonzero(inputs, axis=-1, dtype=tf.int32), 0)
        loss = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), mask)
        loss = tf.reduce_mean(loss) 
        
    with tf.variable_scope("train"):
        train_op = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(loss) 
    
    with tf.variable_scope("accuracy"):
        correct_pred = tf.boolean_mask(tf.equal(prediction, labels), mask)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return inputs, labels, input_prob, output_prob, state_prob, sequence_length, loss, train_op, acc