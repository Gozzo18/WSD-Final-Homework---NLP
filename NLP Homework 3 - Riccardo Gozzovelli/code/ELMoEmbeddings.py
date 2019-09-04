import pickle
import tensorflow_hub as hub


def ELMo_module(embedding_file, length_group):
    """
    Initiliaze the ELMo module, generate the embeddings and retrieve them by reading the pickle file created in the function elmo_vectors

    :param embedding_file: path of the embedding_file
    :param lenght_group: list of sentences grouped by length
    :return elmo_model: dictionary embedding, key = word, value = embedding with size 512
    """
    #elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    #print(elmo)
    #for group in length_group:
    #    list_sentences =  [' '.join(x) for x in group]
    #    elmo_vectors(list_sentences, embedding_file, elmo)
            
    with open(embedding_file, 'rb') as f:
            pickle_objects = []
            while 1:
                    try:
                            single_object = pickle.load(f)
                    except EOFError:
                            break
                    pickle_objects.append(single_object)
        
    elmo_model = {}
    elmo_model['PAD'] = 0.0
    for obj in pickle_objects:    
            for token in obj:
                key = token
                value = obj[token]
                if not token in elmo_model:
                        elmo_model[key] = value
    #print("Length of ", word, " embeddings vocabulary: ", len(elmo_model), "\n")
    
    return elmo_model

def elmo_vectors(batch_sentences, embedding_file, elmo):
    """
    Generate embeddings using ELMo module of tensorflow. Words and their embeddings are inserted into a dictionary which is then saved in memory.
    To avoid auto-padding of the sentence, batch_sentences contains elements of the same length.
    
    :param batch_sentences: list of sentences with the same length
    :param embedding_file: name of the file where to save the dictionary
    """
    vocab = {}
    embeddings = elmo(batch_sentences, signature="default", as_dict=True)["elmo"]
    print(embeddings.shape)
    
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            #Generate the embeddings
            embeddings = sess.run(embeddings)
    
    for sentence in batch_sentences:
            #Get the index of the current sentence
            row_index = batch_sentences.index(sentence)
            for token in sentence.split():
                #Get the index of the current token of the row_index-th sentence
                column_index = (sentence.split()).index(token)
                #Check wheter the token was already added
                if not token in vocab:
                        #If not, extract the embeddings for that token
                        vocab[token] = embeddings[row_index][column_index]
                else:
                        #Otherwhise, check if its embeddings has been updated or not
                        if not vocab[token].any() == embeddings[row_index][column_index].any():
                            vocab[token] =  embeddings[row_index][column_index]
           
    #Save the dictionary in a .pickle file   
    if os.path.exists(embedding_file):
            append_write = 'ab' # append if already exists
    else:
            append_write = 'wb' # make a new file if not
    with open(embedding_file, append_write) as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)