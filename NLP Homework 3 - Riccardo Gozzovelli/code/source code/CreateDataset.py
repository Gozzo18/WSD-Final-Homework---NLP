import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def padding(sequence, length, embedding_size):
    """
    Pad every element of "sequence" to a total of "length" components.
    Padding is done by creating first a numpy array of the correct size with zeros and then copying the element in it.
    
    :param sequence: numpy tensor of sentence embeddings with shape (num, ?, 1024)
    :param length: maximum length of padding
    :param embedding: maximum size of the embedding

    :return padded_x: numpy tensor of padded embeddings with shape (num, length, embedding) 
    """
    sequence = np.asarray(sequence)
    padded_x = np.zeros([sequence.shape[0], length, sequence.shape[2]], dtype='float32')
    padded_x[:sequence.shape[0], :sequence.shape[1], :sequence.shape[2]] = sequence[:, :length, :]
    
    return padded_x[:, :, :embedding_size]

def padDatasets(embedding_file, max_len, embedding_size, padded_sequence_file):
    """
    Recover the saved embeddings, pad them to the given size, and save the padded embedding generated.

    :param embedding_file: path of the file containing the full sentence embeddings
    :param max_len: maximum length of padding
    :param embedding_size: maximum length of the embedding
    :param padded_sequence_file: path of the file containing the padded sentence embeddings

    :return x: numpy tensor of padded sentence embeddings with shape (num_sentences, max_len, embedding_size)

    """

    x = []
    
    #If the padded embeddings are not present, creat ethem
    if not os.path.exists(padded_sequence_file):
        print("Padded dataset is not present. Creating it now.")
        #Retrieve the sentence embeddings from the .pickle file
        with open(embedding_file, 'rb') as f:
            pickle_objects = []
            while 1:     
                try:
                    single_object = pickle.load(f)
                except EOFError:
                     break
                pickle_objects.append(single_object) 
        #Pad each sentence embedding inside the pickle object to max_len
        for obj in pickle_objects:
            padded_sequence = padding(obj, max_len, embedding_size)
            #Save each padded object into another .pickle file   
            if os.path.exists(padded_sequence_file):
                append_write = 'ab' # Append if already exists
            else:
                append_write = 'wb' # Make new file if not
            with open(padded_sequence_file, append_write) as handle:
                pickle.dump(padded_sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Padded dataset is present. Loading it now.")

    #Then retrieve the padded sentences
    with open(padded_sequence_file, 'rb') as f:
        sequence_pickle_objects = []
        while 1:     
            try:
                sequence_single_object = pickle.load(f)
            except EOFError:
                 break
            sequence_pickle_objects.append(sequence_single_object)
            
    for obj in sequence_pickle_objects:
        for token in obj:
            x.append(token)
            
    return np.asarray(x)

def singleTaskTrainingSet(labels, output_vocab, max_length):

    """
    Substitute each label with its own integer value contained in the vocabulary and apply padding.
    
    :param labels: list of tokenized labels
    :param output_vocab: dictionary containing the mapping label => integer
    :param max_length: maximum length of each sample 

    :return y: numpy array of integers
    :return sequence_length: numpy array containing the original length of every sample
    """

    y = []
    sequence_length = []
            
    intermediate_y = []
    for sequence_label in labels:
        length = len(sequence_label)
        if length > max_length:
            length = max_length
        sequence_length.append(length)
        temp = list()
        for label in sequence_label:
            value = output_vocab[label]
            temp.append(value)
        intermediate_y.append(temp) 

    #Pad intermediate_x and intermediate_y to same dimension
    y = pad_sequences(intermediate_y, truncating='post', padding='post', dtype='int64', maxlen=max_length, value=0)

    return np.asarray(y), np.asarray(sequence_length)