import parseDataset
import os
import tensorflow as tf
import numpy as np
import mappings


def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    test_x, token_positions_to_predict = parseDataset.createDataset(input_path, resources_path + 'embeddings/senseval3_embeddings.pickle')
    print()
    print("Building the model and loading weights \n")
    #Build model and load weights
    predictions = []
    tf.reset_default_graph()

    with tf.Session() as sess:
            # Restore variables from disk.    
            saver = tf.train.import_meta_graph(resources_path + "model/checkpoint/model.ckpt.meta")
            graph = tf.get_default_graph()
            saver.restore(sess, resources_path + "model/checkpoint/model.ckpt")
            print("Model restored \n")
            #Get the input tensor
            inputs = graph.get_tensor_by_name('Inputs:0')
            keep_prob = graph.get_tensor_by_name("Probabilities:0")
            #Get the tensor layer after softmax and argmax operation
            prediction_tensor = graph.get_tensor_by_name("dense_sensekeys/ArgMax_Predictions_sensekey:0")
            #Feed the model with one sample at the time
            for sample in test_x[:2]:                         
                    resized_sample = np.reshape(sample, (1, sample.shape[0], 512))
                    padded_sample = np.ones([37, 70, 512], dtype='float32')
                    padded_sample[:resized_sample.shape[0], :resized_sample.shape[1], :resized_sample.shape[2]] = resized_sample 
                    feed_dict = {keep_prob:1.0, inputs:padded_sample}
                    predictions.append(sess.run(prediction_tensor[0], feed_dict))
    predictions = np.asarray(predictions)
    print("Predictions obtained: ", predictions.shape)
    print()
    inverted_vocab = mappings.extractVocabulary(resources_path + 'Mapping_Files/inverted_sensekey_mapping.txt')
    decoded_predictions = mappings.decodeLabels(predictions, inverted_vocab)
    
    #Transform first sensekeys into wordnet synsets
    wordnet_predictions = mappings.sensekeyToWN(decoded_predictions)
    print("Retrieve BabelNet synset \n")
    #Transform wordnet synsets into babelnet synsets
    babelnet_predictions = mappings.wordnetToBN(wordnet_predictions, resources_path + 'Mapping_Files/babelnet2wordnet.tsv')
    
    print("Writing BabelNet predictions file")
    mappings.writePredictionFile(token_positions_to_predict, babelnet_predictions, output_path) 


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    test_x, token_positions_to_predict = parseDataset.createDataset(input_path, resources_path + 'embeddings/senseval3_embeddings.pickle')
    print()
    print("Building the model and loading weights \n")
    #Build model and load weights
    predictions = []
    tf.reset_default_graph()

    with tf.Session() as sess:
            # Restore variables from disk.    
            saver = tf.train.import_meta_graph(resources_path + "model/checkpoint/model.ckpt.meta")
            graph = tf.get_default_graph()
            saver.restore(sess, resources_path + "model/checkpoint/model.ckpt")
            print("Model restored \n")
            #Get the input tensor
            inputs = graph.get_tensor_by_name('Inputs:0')
            keep_prob = graph.get_tensor_by_name("Probabilities:0")
            #Get the tensor layer after softmax and argmax operation
            prediction_tensor = graph.get_tensor_by_name("dense_sensekeys/ArgMax_Predictions_sensekey:0")
            #Feed the model with one sample at the time
            for sample in test_x[:1]:           
                    resized_sample = np.reshape(sample, (1, sample.shape[0], 512))
                    padded_sample = np.ones([37, 70, 512], dtype='float32')
                    padded_sample[:resized_sample.shape[0], :resized_sample.shape[1], :resized_sample.shape[2]] = resized_sample 
                    feed_dict = {keep_prob:1.0, inputs:padded_sample}
                    predictions.append(sess.run(prediction_tensor[0], feed_dict))
    predictions = np.asarray(predictions)
    print("Predictions obtained: ", predictions.shape)
    print()
    inverted_vocab = mappings.extractVocabulary(resources_path + 'Mapping_Files/inverted_sensekey_mapping.txt')
    decoded_predictions = mappings.decodeLabels(predictions, inverted_vocab)
    
    #Transform first sensekeys into wordnet synsets
    wordnet_predictions = mappings.sensekeyToWN(decoded_predictions)
    #Transform wordnet synsets into babelnet synsets
    babelnet_predictions = mappings.wordnetToBN(wordnet_predictions, resources_path + 'Mapping_Files/babelnet2wordnet.tsv')
    print("Retrieve WordNet Domains \n")
    domain_predictions = mappings.bnToDomains(babelnet_predictions, resources_path + 'Mapping_Files/babelnet2wndomains.tsv')
    print("Writing Domain predictions file")
    mappings.writePredictionFile(token_positions_to_predict, domain_predictions, output_path)  


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    test_x, token_positions_to_predict = parseDataset.createDataset(input_path, resources_path + 'embeddings/senseval3_embeddings.pickle')
    print()
    print("Building the model and loading weights \n")
    #Build model and load weights
    predictions = []
    tf.reset_default_graph()

    with tf.Session() as sess:
            # Restore variables from disk.    
            saver = tf.train.import_meta_graph(resources_path + "model/checkpoint/model.ckpt.meta")
            graph = tf.get_default_graph()
            saver.restore(sess, resources_path + "model/checkpoint/model.ckpt")
            print("Model restored \n")
            #Get the input tensor
            inputs = graph.get_tensor_by_name('Inputs:0')
            keep_prob = graph.get_tensor_by_name("Probabilities:0")
            #Get the tensor layer after softmax and argmax operation
            prediction_tensor = graph.get_tensor_by_name("dense_sensekeys/ArgMax_Predictions_sensekey:0")
            #Feed the model with one sample at the time
            for sample in test_x[:1]:           
                    resized_sample = np.reshape(sample, (1, sample.shape[0], 512))
                    padded_sample = np.ones([37, 70, 512], dtype='float32')
                    padded_sample[:resized_sample.shape[0], :resized_sample.shape[1], :resized_sample.shape[2]] = resized_sample 
                    feed_dict = {keep_prob:1.0, inputs:padded_sample}
                    predictions.append(sess.run(prediction_tensor[0], feed_dict))
    predictions = np.asarray(predictions)
    print("Predictions obtained: ", predictions.shape)
    print("Predictions obtained: ", predictions.shape)
    print()
    inverted_vocab = mappings.extractVocabulary(resources_path + 'Mapping_Files/inverted_sensekey_mapping.txt')
    decoded_predictions = mappings.decodeLabels(predictions, inverted_vocab)
    
    #Transform first sensekeys into wordnet synsets
    wordnet_predictions = mappings.sensekeyToWN(decoded_predictions)
    #Transform wordnet synsets into babelnet synsets
    babelnet_predictions = mappings.wordnetToBN(wordnet_predictions, resources_path + 'Mapping_Files/babelnet2wordnet.tsv')
    print("Retrieve Lexicographer prediction \n")
    lex_predictions = mappings.bnToLexnames(babelnet_predictions, resources_path + 'Mapping_Files/babelnet2lexnames.tsv')
    print("Writing Lexicographer predictions file")
    mappings.writePredictionFile(token_positions_to_predict, lex_predictions, output_path)   
