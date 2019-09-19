import re
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

def decodeLabels(predictions, inverted_vocab):
    """
    Map back prediction to word, sensekeys and hypernymy relationships
    
    :param predictions: list of predictions as integers
    :param inverted_vocab: dictionary for inverting the mapping, key = integer, value = word/sensekey/hypernymy
    :return decoded_labels: list of labels as word/sensekey/hypernymy
    """
    decoded_labels =[]
    for sequence in predictions:
        temp = []
        #for word in list(sequence):
        for label in list(sequence):
            if not label == 0:
                if label in inverted_vocab:
                    temp.append(inverted_vocab[label])
        decoded_labels.append(temp)
    
    return decoded_labels

def sensekeyToWN(predictions):
    """
    Retreive wordnet synsets from sensekey

    :param predictions: list of predictions as sensekey
    :return wn_predictions: list of predictions as wordnet synset
    """
    wn_predictions = []
    sensekey_regex_identifier = r'(%[1-9])'
    for sequence in predictions:
            temp = []
            for token in sequence:
                    if re.search(sensekey_regex_identifier, token):
                        pos = wn.lemma_from_key(token).synset().pos()
                        offset = wn.lemma_from_key(token).synset().offset()
                        token = "wn:" + str(offset).zfill( 8) + pos
                    temp.append(token)
            wn_predictions.append(temp)
    return wn_predictions
    
def wordnetToBN(predictions, resource_mapping):
    """
    Retreive babelnet synsets from wordnet synsets

    :param predictions: list of predictions as wordnet synsets
    :param resource_mapping: path of the file containing the mapping between wn synsets and bn synsets
    :return bn_predictions: list of predictions as babelnet synset
    """
    wn_to_bn = {}
    with open(resource_mapping, 'r') as mapping_file:
            for line in mapping_file:
                    bn, wn = line.split()
                    if not wn in wn_to_bn:
                            wn_to_bn[wn] = [bn]
                    else:
                            wn_to_bn[wn].append(bn)
    bn_predictions = []
    for sequence in predictions:
            temp = []
            for token in sequence:
                    if 'wn:' in token:
                            token = wn_to_bn[token][0]
                    temp.append(token)       
            bn_predictions.append(temp)
        
    return bn_predictions
    
def bnToDomains(predictions, resource_mapping):
    """
    Retreive wordnet domains from babelnet synsets

    :param predictions: list of predictions as babelnet synsets
    :param resource_mapping: path of the file containing the mapping between bn synsets and wn domains
    :return domain_predictions: list of predictions as wordnet domains
    """
    bn_to_dom = {}
    with open(resource_mapping, 'r') as mapping_file:
            for line in mapping_file:
                    bn = line.split()[0]
                    dom = line.split()[1]
                    if not bn in bn_to_dom:
                            bn_to_dom[bn] = [dom]
                    else:
                            bn_to_dom[bn].append(dom)
    domain_predictions = []
    for sequence in predictions:
            temp = []
            for token in sequence:
                    if 'bn:' in token:
                            if token in bn_to_dom:
                                    token = bn_to_dom[token][0]
                            else:
                                    token = 'factotum'
                    temp.append(token)       
            domain_predictions.append(temp)
        
    return domain_predictions
    
    
def bnToLexnames(predictions, resource_mapping):
    """
    Retreive wordnet lexnames from babelnet synsets

    :param predictions: list of predictions as babelnet synsets
    :param resource_mapping: path of the file containing the mapping between bn synsets and wn lexnames
    :return lexname_predictions: list of predictions as lexnames
    """
    bn_to_lex = {}
    with open(resource_mapping, 'r') as mapping_file:
            for line in mapping_file:
                    bn = line.split()[0]
                    lex = line.split()[1]
                    if not bn in bn_to_lex:
                            bn_to_lex[bn] = [lex]
                    else:
                            bn_to_lex[bn].append(lex)
    lexname_predictions = []
    for sequence in predictions:
            temp = []
            for token in sequence:
                    if 'bn:' in token:
                            token = bn_to_lex[token][0]
                    temp.append(token)       
            lexname_predictions.append(temp)
        
    return lexname_predictions

def writePredictionFile(prediction_positions, predictions, output_file):
    """
    Write prediction file

    :param prediction_positions: list of integers, representing the position of each token in the given sentence to disambiguate
    :param predictions: list of predictions
    :param outuput_file: path of the predicted file
    """
    with open(output_file, 'w') as prediction_file:
        for i in range(len(prediction_positions)):
            label_index = 0
            for j in range(len(prediction_positions[i])):
                if not prediction_positions[i][j] == '0':
                        prediction = predictions[i][label_index]
                        line = prediction_positions[i][j] + ' ' + prediction + '\n'
                        prediction_file.write(line)     
                        label_index += 1