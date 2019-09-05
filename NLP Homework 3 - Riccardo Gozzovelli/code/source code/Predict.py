import os
from lxml import etree
import re
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
import tensorflow_hub as hub
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
import csv
import itertools

def extractTokensToPredict(data_path):
    """
    Parse the dataset and extract sentences and pointers to only those tokens that need to be disambiguated.
    They are identified by the 'id' attribute

    :param data_path: dataset file name
    :return sentences: list of tokenized sentences
    :return ids: list of identifier for lemmas to predict
    """
    
    sentences = []
    ids = []
    
    for event, element in etree.iterparse(data_path, tag="sentence"):
        current_sentence = []
        current_ids = []
        if event == 'end':
            #For every child of the sentence tag
            for child in element:
                #Get the lemma of the token
                lemma = child.attrib['lemma']
                if '&apos;' in lemma:
                    #If it is present, substitute it
                    lemma = re.sub(r'(&apos;)', '\'', lemma)
                    #Check also for &apos;&apos = ""
                    if '\'\'' in word:
                        lemma = re.sub(r'(\'\')', '\'', lemma)
                if child.tag == 'instance':
                    current_ids.append(child.attrib['id'])
                else:
                    current_ids.append('0')
                current_sentence.append(lemma)
        if current_sentence and current_ids:
            sentences.append(current_sentence)
            ids.append(current_ids)
        #Clear to save memory
        element.clear()
    
    print("File completely parsed. Total number of sentences %i \n" %len(sentences))
    print()
    return sentences, ids

def extractLabelIdentifier(sentences, ids, lemmas_mapping, vocab_identifier, wordnetCompression):
    """
    Every lemma to disambiguate is associated to a label which is mapped to an integer.
    This mapping is contained in the vocab_identifier variable- Lemmas to disambiguate are first searched in the "lemmas_mapping" dictionary.
    If they are not present there the MSF (most frequent sense) method is applied. Thus their label is recovered from the WordNet interface.
    
    :param sentences: list of tokenized sentences
    :param ids: list of identifier for lemmas to predict
    :param lemmas_mapping: lemma => label mapping
    :param vocab_identifier: label => integer mapping
    :param wordnetCompression: boolean variable used to check if a model with the wordnet synset compression method has been used
    :return identifiers_list: list of list of integers and sensekey; each internal list correspond to all the sensekey identifier associated to a lemma
    """

    identifiers_list = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        id_sequence = ids[i]
        sentence_ids = []
        for j in range(len(sentence)):
            word = sentence[j]
            id = id_sequence[j]
            word_ids = []
            #Check if the current word was met during training
            if not id == '0':
                if word in lemmas_mapping:
                    #If it is, ectract the sensekeys associated to the lemma
                    sensekeys = lemmas_mapping[word]
                    #Then search for all the sensekeys their identifier
                    for sensekey in sensekeys:
                        word_ids.append(vocab_identifier[sensekey])
                else:
                    #Take the most frequent sense from wordnet
                    mfs = str(wn.synsets(word)[0])[8:-2]
                    #Retrieve the correspondent sensekey
                    sensekey = wn.synset(mfs).lemmas()[0].key()
                    if wordnetCompression:
                        #Transform the senekey into a wordnet synset
                        pos = wn.lemma_from_key(sensekey).synset().pos()
                        offset = wn.lemma_from_key(sensekey).synset().offset()
                        wn_synset = "wn:" + str(offset).zfill( 8) + pos
                        word_ids.append(wn_synset)
                    else:
                        word_ids.append(sensekey)
            if word_ids:
                sentence_ids.append(word_ids)
        identifiers_list.append(sentence_ids)

    return identifiers_list

def ELMo_module(tokenized_sentences, embedding_size): 
    """
    Slightly different method than the one contained in the ELMo class.
    Here we generate the embeddings for the test set all togheters
    
    """

    sequence_length = []
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    for sentence in tokenized_sentences:
        sequence_length.append(len(sentence))
    list_sentences =  [' '.join(x) for x in tokenized_sentences]
    sentence_embedding = elmo_vector(list_sentences, elmo)

    return sequence_length, sentence_embedding[:,:,:embedding_size]

def elmo_vector(sentence, elmo):
    embeddings = elmo(sentence, signature="default", as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embeddings = sess.run(embeddings)
    
    return embeddings


def predictions_architecture_1_2(test_x, sequence_length, complete_model, graph_path):
    """
    Restore the trained model and feed it with the test set to generate predictions

    :param test_x: 3d numpy array of containing the embeddings of the sentences
    :param sequence_length: 1d numpy array containing the original length of every sample
    :return predictions: 3d tensor of prediction for every sentence
    
    """
    print("Predictions")
    predictions = []
    tf.reset_default_graph()
    with tf.Session() as sess:
            # Restore variables from disk.    
            saver = tf.train.import_meta_graph(graph_path)
            graph = tf.get_default_graph()
            saver.restore(sess, complete_model)
            print("Model restored \n")
            #Get the required tensor to start predictions
            inputs = graph.get_tensor_by_name('Inputs:0')
            input_prob = graph.get_tensor_by_name("Input_Probability:0")
            output_prob = graph.get_tensor_by_name("Output_Probability:0")
            state_prob = graph.get_tensor_by_name("State_Probability:0")
            seq_length = graph.get_tensor_by_name("Sequence_Length:0")
            #Get the softmax layer
            prediction_tensor = graph.get_tensor_by_name("dense/Softmax_Logits:0")
            #Feed the model with one sample at the time
            for i in range(len(test_x)): 
                    if i % 20 == 0:
                        print(i)
                    #Reshape the sample so as to have the correct shape
                    sample = np.reshape(test_x[i], (1, test_x.shape[1], test_x.shape[2] ))
                    #Copy the sample into a smaller 3d numpy array where the second dimension is the correct one
                    padded_x = np.zeros([1, sequence_length[i], 400], dtype='float32')
                    padded_x[:sample.shape[0], :sequence_length[i], :sample.shape[2]] = sample[:,:sequence_length[i],:]
                    #Feed the model
                    feed_dict = {input_prob:1.0, output_prob:1.0, state_prob:1.0, inputs:padded_x, seq_length:[sequence_length[i]]}
                    predictions.append(sess.run(prediction_tensor[0], feed_dict))
    predictions = np.asarray(predictions)
    print("Predictions obtained: ", predictions.shape)
    print()
    return predictions

def evaluate(predictions, ids, label_identifiers):
    """
    Predictions have shape (number_of_sentences, length, output_vocabulary).
    Model evaluation is done by targeting only the indexes of the sensekeys of the lemma to disambiguate.
    We therefore apply the argmax operation only on some values for every lemma.

    :param predictions: 3d numpy array containing the predictions for every sample
    :return labels: list of integer corresponding to the sensekeys predicted for every sample
    
    """

    labels = []
    #For every prediction
    for i in range(len(predictions)):
        sentence_predictions = predictions[i]
        id_sequence = ids[i]
        sequence_labels = []
        counter = 0
        #For every predicted token
        for j in range(len(id_sequence)):
            word_prediction = sentence_predictions[j]
            id = id_sequence[j]
            #Take only the lemmas that have to be disambiguated
            if not id == '0':
                #Extract the identifiers of the sensekeys associated to the lemma
                indexes = label_identifiers[i][counter]
                new_predictions = []
                #Check if the identifier is a number            
                for elem in indexes:
                    try:
                        index = int(elem)
                        new_predictions.append(predictions[i][j][index])
                    except ValueError:
                        #If is not, MFS was applied
                        new_predictions.append(elem)
                #Do the argmax on the extracted prediction indexes
                argmax = np.argmax(new_predictions)
                label = label_identifiers[i][counter][argmax]
                sequence_labels.append(label)
                counter += 1
        labels.append(sequence_labels)

    return labels

def decodeLabels(predictions, vocab):
    """
    Use the inverse mapping in vocab to transform predictions with identifiers to predicitons with labels

    :param predictions: list of integers representing the predicted identifiers
    :param vocab: dictionary containing the mapping integer => sensekeys
    """

    decoded_labels =[]
    for sequence in predictions:
        temp = []
        for label in list(sequence):
            if label in vocab:
                temp.append(vocab[label])
            else:
                temp.append(label)
        decoded_labels.append(temp)
    
    return decoded_labels

def F1Score(labels, gold_file, wordnetCompression):
    """
    Compute the F1 score

    :param labels: list of labels predicted by the model
    :param gold_file: path of the file containing the correct labels
    """

    gold_labels = []
    with open(gold_file, 'r') as file:
        for line in file:
            gold_labels.append(line.split()[1])

    flat_decoded_label_list = [item for sublist in labels for item in sublist]

    total = 0
    correct = 0
    if wordnetCompression:
        for i in range(len(gold_labels)):
            total += 1
            current_gold = gold_labels[i]
            pos = wn.lemma_from_key(current_gold).synset().pos()
            offset = wn.lemma_from_key(current_gold).synset().offset()
            wn_synset = "wn:" + str(offset).zfill( 8) + pos
            current_prediction = flat_decoded_label_list[i]
            #if len(current_prediction) > 1:
            #    for elem in current_prediction:
            #        if elem == current_gold:
            #            correct += 1
            #else:
            if wn_synset == current_prediction:
                    correct += 1
    else:
        for i in range(len(gold_labels)):
            if gold_labels[i] == flat_decoded_label_list[i]:
                correct += 1
            total += 1
        
    print("F1 SCORE: ", (correct/total)*100 )
    print("Total number of labels: ", total)
    print("Total correct: ", correct)
    print()
