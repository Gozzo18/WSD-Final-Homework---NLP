import re
import csv
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

def extractMappings(file_list):
    """
    Extract the mapping from the given list of file.

    :param file_list: list of file name from which extract the mapping
    :return wordNet_to_babelNet: dictionary containing the mapping wn_synset => bn_synset
    :return babelNet_to_wnDomain: dictionary containing the mapping bn_synset => wn domain
    :return babelNet_to_lexNames: dictionary containing the mapping bn_synset => lexical name
    """
    
    wordNet_to_babelNet = {}
    babelNet_to_wnDomain = {}
    babelNet_to_lexNames = {}

    for mapping_file in file_list:
        if 'wordnet' in mapping_file:
            mappingToWN = True
        else:
            mappingToWN = False
            if 'domains' in mapping_file:
                mappingToDomains = True
            else:
                mappingToDomains = False
        with open(mapping_file, 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                if mappingToWN:
                    key = line[1]
                    value = line[0]
                    wordNet_to_babelNet[key] = value
                else:
                    if len(line)>2:
                        value = '.'.join(line[1:])
                    else:
                        value = line[1]
                    key = line[0]
                    if mappingToDomains:
                        babelNet_to_wnDomain[key] = value
                    else:
                        babelNet_to_lexNames[key] = value
                                        
    return wordNet_to_babelNet, babelNet_to_wnDomain, babelNet_to_lexNames

def wnToBn(wn_labels, wordNet_to_babelNet, wordnetCompression):
    """
    Retrieve the list of labels containing BabelNet synsets

    :param wn_labels: list of labels with wordnet synsets
    :param  wordNet_to_babelNet: dictionary containing the mapping wn_synset => bn_synset
    :param wordnetCompression: boolean variable used to check whether is required to pass before from sensekey => wn_synsets and then to wn_synset => bn_synset or not.
    :return bn_label_synsets: list of labels with babelnet synsets
    
    """

    bn_label_synsets = []
    for label_sequence in wn_labels:
        temp = []
        for label in label_sequence:
            if wordnetCompression:
                if 'wn:' in label:
                    temp.append(wordNet_to_babelNet[label])
                else:
                    temp.append(label)
            else:
                if re.search(r'(%[1-9])', label):
                    pos = wn.lemma_from_key(label).synset().pos()
                    offset = wn.lemma_from_key(label).synset().offset()
                    synset_id = "wn:" + str(offset).zfill( 8) + pos
                    temp.append(wordNet_to_babelNet[synset_id])
                else:
                    temp.append(label)
        bn_label_synsets.append(temp)

    return bn_label_synsets
   
def bnToWnDomain(bn_labels, babelNet_to_wnDomain):
    """
    Retrieve the list of labels containing wordnet domains

    :param bn_labels: list of labels with babelnet synsets
    :param  babelNet_to_wnDomain: dictionary containing the mapping bn_synset => wordnet domain
    :return domain_labels: list of labels with wordnet domains
    
    """

    domain_labels = []
    for label_sequence in bn_labels:
        temp = []
        for label in label_sequence:
            if 'bn:' in label:
                if label in babelNet_to_wnDomain:
                    temp.append(babelNet_to_wnDomain[label])
                else:
                    temp.append('factotum')
            else:
                temp.append(label)
        domain_labels.append(temp)
    
    return domain_labels

def bnToWnLex(bn_labels, babelNet_to_lexNames):
    """
    Retrieve the list of labels containing lexnames

    :param bn_labels: list of labels with babelnet synsets
    :param  babelNet_to_lexNames: dictionary containing the mapping bn_synset => lexical names
    :return lex_labels: list of labels with lexical names
    
    """

    lex_labels = []
    for label_sequence in bn_labels:
        temp = []
        for label in label_sequence:
            if 'bn:' in label:
                temp.append(babelNet_to_lexNames[label])
            else:
                temp.append(label)
        lex_labels.append(temp)
    
    return lex_labels
