import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
import tensorflow_hub as hub
import os 
import pickle
import numpy as np

def elmo_vectors(batch_sentences, embedding_file, elmo):
    """
    Generate sentence embeddings and save them in a .pickle file.
    
    :param batch_sentences: list of sentences with same length
    :param embedding_file: path of the file where to save embeddings

    """
    
    vocab = {}
    embeddings = elmo(batch_sentences, signature="default", as_dict=True)["elmo"]
    print(embeddings.shape)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        #Generate the embeddings
        embeddings = sess.run(embeddings)

    #Save the sentence embeddings
    if os.path.exists(embedding_file):
        append_write = 'ab' # Append if already exists
    else:
        append_write = 'wb' # Create new file if first time
    with open(embedding_file, append_write) as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def Module(embedding_file, length_group):
    """
    Check if the embeddings are already present. If not initialize the elmo module and start producing embeddings for each sentence.
    Sentences are fed to the ELMo neural network in groups where sentences share the same length. 
    This prevents ELMo from truncating/padding sentence with different lengths. 

    :param embedding_file: path of the file where to save embeddings
    :param length_group: list of list of sentence. Each internal list contains tokenized sentences with the same length

    """

    if not os.path.exists(embedding_file):
        print("Embeddings are not present. Creating them now.")
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        print(elmo)
        for group in length_group:
            list_sentences =  [' '.join(x) for x in group]
            elmo_vectors(list_sentences, embedding_file, elmo)
    else:
        print("Embeddings are present. Loading them now.")


    