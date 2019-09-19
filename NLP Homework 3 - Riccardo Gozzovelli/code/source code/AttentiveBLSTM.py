import tensorflow as tf

def multitaskBidirectionalModel(batch_size, embedding_size, hidden_size, padding_length, output_vocabulary_length, domain_vocabulary_length, lexname_vocabulary_length):
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, embedding_size], name="Inputs")
    print("INPUT TENSOR SIZE ", inputs)
             
    sensekey_labels = tf.placeholder(tf.int64, shape=[None, None], name='Sensekey_Labels')
    print("LABEL TENSOR SIZE ", sensekey_labels)
    
    domain_labels = tf.placeholder(tf.int64, shape=[None, None], name='Domain_Labels')
    print("DOMAIN LABELS TENSOR SIZE ", domain_labels)
    
    lexname_labels = tf.placeholder(tf.int64, shape=[None, None], name='Lexnames_Labels')
    print("LEXNAMES LABELS TENSOR SIZE ", lexname_labels)
      
    keep_prob = tf.placeholder(tf.float32, shape=[], name='Probabilities')
    print("KEEP PROB. TENSOR ", keep_prob)
    
    lambda_1 = tf.placeholder(tf.float32, shape=[], name='Lambda_1')
    print("LAMBDA_1 TENSOR ", lambda_1)
    
    lambda_2 = tf.placeholder(tf.float32, shape=[], name='Lambda_2')
    print("LAMBDA_2 TENSOR ", lambda_2)

    sequence_length = tf.placeholder(tf.int32, shape=[None], name="Sequence_Length")
    print("SEQUENCE_LENGTH TENSOR ", sequence_length)
    
    with tf.variable_scope("rnn_1"):
        rnn_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell,
                                            input_keep_prob=keep_prob,
                                            output_keep_prob=keep_prob,
                                            state_keep_prob=keep_prob)
        rnn_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell,
                                            input_keep_prob=keep_prob,
                                            output_keep_prob=keep_prob,
                                            state_keep_prob=keep_prob)
        
        (forward_output_1, backward_output_1), states_1 = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, inputs,  sequence_length=sequence_length, dtype='float32')
        outputs_1 = tf.concat([forward_output_1, backward_output_1], axis=2)
        
    with tf.variable_scope("rnn_2"):
        rnn_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell,
                                            input_keep_prob=keep_prob,
                                            output_keep_prob=keep_prob,
                                            state_keep_prob=keep_prob)
        rnn_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell,
                                            input_keep_prob=keep_prob,
                                            output_keep_prob=keep_prob,
                                            state_keep_prob=keep_prob)
        
        (forward_output_2, backward_output_2), states_2 = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, outputs_1, sequence_length=sequence_length, dtype='float32')
        outputs_2 = tf.concat([forward_output_2, backward_output_2], axis=2)

    mask = tf.greater(tf.count_nonzero(inputs, axis=-1, dtype=tf.int32), 0)
                
    def attention(input_x, W_att):
        h_tanh = tf.tanh(input_x)
        u = tf.matmul(h_tanh, W_att)
        a = tf.nn.softmax(u)
        c =  tf.multiply(h_tanh, a)
        return c 
    
    with tf.variable_scope("attention"):
        omega = tf.get_variable("omega_att_local", shape=[2*hidden_size, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=0))
        c = attention(outputs_2, omega)
        attention_plus_output = tf.concat([outputs_2, c], -1)    

    with tf.variable_scope("dense_sensekeys"):
        unscaled_s_logits = tf.layers.dense(attention_plus_output, output_vocabulary_length, activation=None)
        sensekey_logits = tf.nn.softmax(unscaled_s_logits, name='Sensekey_Logits')
        sensekey_predictions = tf.argmax(sensekey_logits, -1)
    
    with tf.variable_scope("dense_domains"):
        unscaled_d_logits = tf.layers.dense(attention_plus_output, domain_vocabulary_length, activation=None)
        domain_logits = tf.nn.softmax(unscaled_d_logits, name='Domain_Logits')
        domain_predictions = tf.argmax(domain_logits, -1) 
    
    with tf.variable_scope("dense_lexnames"):
        unscaled_l_logits = tf.layers.dense(attention_plus_output, lexname_vocabulary_length, activation=None)
        lexname_logits = tf.nn.softmax(unscaled_l_logits, name='Lexname_Logits')
        lexname_predictions = tf.argmax(lexname_logits, -1) 

    with tf.variable_scope("main_loss"):
        main_loss = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sensekey_labels, logits=unscaled_s_logits), mask)
        main_loss = tf.reduce_mean(main_loss) 
        
    with tf.variable_scope("auxiliary_loss_1"):
        aux_loss_1 = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=domain_labels, logits=unscaled_d_logits), mask)
        aux_loss_1 = tf.reduce_mean(aux_loss_1) 
    
    with tf.variable_scope("auxiliary_loss_2"):
        aux_loss_2 = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lexname_labels, logits=unscaled_l_logits), mask)
        aux_loss_2 = tf.reduce_mean(aux_loss_2) 
        
    with tf.variable_scope("total_loss"):
        total_loss = main_loss + (lambda_1 * aux_loss_1) + (lambda_2 * aux_loss_2) 
        
    with tf.variable_scope("train"):
        train_op = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(total_loss)

    with tf.variable_scope("accuracy"):
        correct_pred = tf.boolean_mask(tf.equal(sensekey_predictions, sensekey_labels), mask)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return inputs, sensekey_labels, domain_labels, lexname_labels, keep_prob, lambda_1, lambda_2, sequence_length, main_loss, aux_loss_1, aux_loss_2, train_op, acc