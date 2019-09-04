import re
import os
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')


def lemmasToSynsets(sentences, labels, isTrain, wordnetCompression):
    """

    Transform every lemma associated to a sensekey in a wordnet synset. 
    Then add the mapping lemma => wn_synset, wn_synset => sensekey in two dictionaries dictionary that will be saved as a .txt file.
    
    :param sentences: list of sentences
    :param labels: list of labels composed by lemmas and sensekeys
    :param isTrain: boolean variable used to distinguish between training and dev set operations
    :param wordnetCompression: boolean variable used to check whether the wordnet_synset => sensekeys mapping must be saved
    :return updated_sentences = list of sentences with lemmas and wn_synsets
    :return updated_labels = list of labels with lemmas and wn_synsets 
    
    """
    lemma_to_wn = {} #Dictionary containing the mapping lemma => wordnet synset
    wn_to_sensekeys = {} #Dictionary containing the mapping wordnet synset => sensekey
    updated_sentences = []
    updated_labels = []

    #For every sentence of the dataset
    for i in range(len(sentences)):
        temp = []
        current_label_sequence= labels[i]
        current_sentence = sentences[i]
        #For every token in the current sentence
        for j in range(len(current_sentence)):
            lemma = current_sentence[j]
            label = current_label_sequence[j]
            #Check if the label is a sensekey
            if re.search(r'(%[1-9])', label):
                #From the sensekey extract the synset
                pos = wn.lemma_from_key(label).synset().pos()
                offset = wn.lemma_from_key(label).synset().offset()
                wn_synset = "wn:" + str(offset).zfill( 8) + pos
                #Add pair (lemma, wordnet_synset) to the dictionary
                if not lemma in lemma_to_wn:
                    lemma_to_wn[lemma] = [wn_synset]
                else:
                    if not wn_synset in lemma_to_wn[lemma]:
                        lemma_to_wn[lemma].append(wn_synset)
                lemma = wn_synset
            temp.append(lemma)
            #Add pair (wordnet_synset, sensekey) to the vocabulary
            if not lemma in wn_to_sensekeys:
                wn_to_sensekeys[lemma] = [label]
            else:
                if not label in wn_to_sensekeys[lemma]:
                    wn_to_sensekeys[lemma].append(label)
        updated_sentences.append(temp)
        updated_labels.append(temp)

    #If we worked on the training set, save the dictionary into two files
    if isTrain:
        if wordnetCompression:
            if not os.path.exists('../../resource/Mapping_Files/wn_to_sensekeys.txt'):
                with open('../../resource/Mapping_Files/wn_to_sensekeys.txt', 'w') as file:
                    for elem in wn_to_sensekeys:
                        line = elem + " " + " ".join(wn_to_sensekeys[elem])
                        file.write(line + "\n")
            if not os.path.exists("../../resource/Mapping_Files/lemma_to_wn.txt"):
                with open("../../resource/Mapping_Files/lemma_to_wn.txt", 'w') as file:
                    for elem in lemma_to_wn:
                        line = elem + " " + " ".join(lemma_to_wn[elem])
                        file.write(line + "\n")

    return updated_sentences, updated_labels