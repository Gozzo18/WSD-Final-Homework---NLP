import re
import mappings
import ELMoEmbeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences

def createDataset(input_path, embeddings_path):
    """
    Parse the file input_path, generate the embeddings and create the dataset for the model

    :param input_path: path of the file to predict
    :param embeddings_path: path to save the embeddings
    """
    punctuation = '!"#$&\'``()*,-./:;<=>?@[\]^_`{}~'
    sentences = []
    token_positions_to_predict = []
    token_position = 0

    with open(input_path, 'r') as input_file:
            #Start parsing the various tag
            for line in input_file:
                    #If the last tag of the file is not reached
                    if not '</corpus>' == line:
                            #If the tag is <sentence> initialize two lists
                            if '<sentence' in line:
                                    current_sentence = []
                                    temp_token_positions = []
                            #If we finished to parse the sentence, append the two lists to the main ones
                            elif '</sentence' in line: 
                                    if current_sentence:   
                                            sentences.append(current_sentence)
                                            token_positions_to_predict.append(temp_token_positions)
                                            #Reset token_position value
                                            token_position = 0
                            else:
                                    #If the taf is wf or instance, then we extract from them certain informations
                                    if '<wf' in line or '<instance' in line:
                                            #Extract the text attribute of the tag, i.e. everything between > and <
                                            token_regex = r'(?<=\>).*?(?=\<)' 
                                            token = (re.search(token_regex, line)).group(0)
                                            #Check for the special sequence &apos corresponding to '
                                            if '&apos;' in token:
                                                    #If it is present, substitute it
                                                    token = re.sub(r'(&apos;)', '\'', token)
                                                    #Check also for &apos;&apos = ""
                                                    if '\'\'' in token:
                                                            token = re.sub(r'(\'\')', '\'', token)
                                            #Avoid all punctuation tokens
                                            if not token in punctuation:
                                                    #Strip the token in case is a multi-token word
                                                    token = token.replace(' ', '_')
                                                    current_sentence.append( token )
                                                    #If the line is an instance, extract the identifier dxxx.sxxx.txxx
                                                    if '<instance' in line:
                                                            instance_id_regex = r'(?<=\").*?(?=\" lemma)'
                                                            temp_token_positions.append( (re.search(instance_id_regex, line)).group(0) + ' ' + str(token_position))
                                                    token_position += 1
    
    print("File completely parsed. Total number of sentences %i \n" %len(sentences))
    print("Creating embeddings \n")
    
    #Sort sentences in ascending order, in order to ease the embedding computation
    sorted_sentences, _, length_group = sortAndGroup(sentences, None)
    prediction_embeddings = ELMoEmbeddings.ELMo_module(embeddings_path, length_group)
    print("Embeddings created.")
    
    #Create dataset
    intermediate_x = []
    for sentence in sentences:
            temp = []
            for word in sentence:
                    if word in prediction_embeddings:
                            temp.append(prediction_embeddings[word][:512])
            intermediate_x.append(temp)
    test_x = pad_sequences(intermediate_x, truncating='post', padding='post', dtype="float32", maxlen=70, value=0.0)
    print("Dataset extracted. Shape : ", test_x.shape)
    return test_x, token_positions_to_predict

def sortAndGroup(sentences, labels):
    
    """
    Sort the sentence and label list by the length attribute of each element 
    
    :param sentences: list of splitted sentences
    :param labels: list of splitted labels
    :return sorted_sentences: ordered list of splitted sentences
    :return sorted_labels: ordered list of splitted labels
    """
    
    sorted_sentences = sorted(sentences, key=len)
    sorted_labels = []
    if labels:
            sorted_labels = sorted(labels, key=len)
        
    grouped_sentences_vocab={}
    for sentence in sorted_sentences:
            grouped_sentences_vocab.setdefault(len(sentence), []).append(sentence)

    grouped_sentences_by_length = [grouped_sentences_vocab[n] for n in sorted(grouped_sentences_vocab)]
       
    return sorted_sentences, sorted_labels, grouped_sentences_by_length
