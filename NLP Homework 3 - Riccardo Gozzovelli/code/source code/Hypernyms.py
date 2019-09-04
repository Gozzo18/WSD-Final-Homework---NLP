import re
import os
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
import itertools


def get_hypernyms(synset):
    """
    Return all the hypernyms given a wordnet synset
    
    :param synset: wordnet synset
    :return: list of hypernyms as ojects

    """ 
    hypos = lambda s:s.hypernyms()
    return list(synset.closure(hypos))

def ancestors(sense):
    """
    Return all the ancestor of a sense as lemmas.

    :param sense: sensekey
    :return ancestor_list: list of ancestor of the sensekey as strings if present, else None

    """

    ancestor_list = []
    #Retrieve the lemma associated to the sensekey
    lemma = wn.lemma_from_key(sense)
    #Retrieve the wordnet synset
    synset = lemma.synset()
    #Get all the hypernyms
    hypernyms_synset = get_hypernyms(synset)
    #Check whether the sensekey as at least one hypernym
    if not hypernyms_synset:
        return None
    else:
        for elem in hypernyms_synset:
            #Parse the ancestor objects to obtain strings
            ancestor_list.append(str(elem)[8:-2])
        return ancestor_list

def extractHypernymTrees(lemma_to_sensekeys):
    """
    Retrieve all the possible hierarchy trees between all the sensekeys associated to a same lemma

    :param sensekeys: dictionary of lemma => sensekey
    :return hypernyms_hirearchy: dictionary containing all the possible trees for every pair of sensekeys 
    
    """

    hypernyms_hierarchy = {}
    #For every sensekey
    for lemma in lemma_to_sensekeys:
        #For every possible pair of sensekeys
        for sensekey in lemma_to_sensekeys[lemma]:
            #Get the ancestor list
            ancestor_list = ancestors(sensekey)
            if ancestor_list:
                #If not emtpy, add it to the dictionary
                hypernyms_hierarchy[sensekey] = ancestor_list
    return hypernyms_hierarchy

def extractLowestAncestor(hypernyms_hierarchy):
    """
    Retrieve from all the hierarchy trees of a sensekey the lowest ancestor among all of them

    :param hypernyms_hierarchy: dictionary containing the mapping sensekey => list of hierarchy trees
    :return updated_hypernyms_relationships: dictionary containing the mapping sensekey => lowest ancestor
    
    """

    updated_hypernyms_relationships = {}
    #For every sensekey
    for sensekey in hypernyms_hierarchy:
        #Reverse the list. The lowest ancestor is the last element of the original list
        hypernyms_list = (hypernyms_hierarchy[sensekey])[::-1]
        #For every other sensekey 
        for comparison in hypernyms_hierarchy:
            #Reverse again, so that it is possible to compare the hierarchy of "sensekey" with the hierachy of "comparison"
            comparison_list = (hypernyms_hierarchy[comparison])[::-1]
            oneEqual = False
            for i in range(len(hypernyms_list)):
                try:
                    #If the two sensekeys have a common ancestor, they share part of the hypernym tree structure 
                    if hypernyms_list[i] == comparison_list[i]:
                        oneEqual = True
                    #Then search for the first node in the tree which is different between the two sensekey's hierarchies
                    if not hypernyms_list[i] == comparison_list[i] and oneEqual:
                        #If the mapping sensekeys => lowest ancestor was never met, we add it to the dictionary
                        if not sensekey in updated_hypernyms_relationships:
                            updated_hypernyms_relationships[sensekey] = hypernyms_list[i]
                        else:
                            #Otherwise, we first check if the new node is in a lower level in the tree structure w.r.t. the previous added node
                            if i > hypernyms_list.index(updated_hypernyms_relationships[sensekey]):
                                updated_hypernyms_relationships[sensekey] = hypernyms_list[i]
                        break
                #If for a sensekey a matching is not found, simply add as hypernym the lowest ancestor of its hierarchy tree
                except IndexError:
                    if oneEqual:
                        if not sensekey in updated_hypernyms_relationships:
                            updated_hypernyms_relationships[sensekey] = hypernyms_list[i]
    return updated_hypernyms_relationships

def sensekeysToHypernyms(sentences, labels):

    """
    Substitute every sensekey with the lowest ancestor among all the possible hierarchy trees that is possible to obtain between all the sensekeys.
    The lowest ancestor is the very first node of the hypernym hierarchy tree that allows to correctly disambiguate the meaning of two words.

    :param sentences: list of sentences
    :param labels: list of labels with sensekeys
    :return hypernym_labels: updated list of labels with hypernyms
    """

    lemma_to_sensekeys = {}
    #If the following path does not exist, extract the mapping
    if not os.path.exists("../../resource/Mapping_Files/lemma_to_sensekeys.txt"):
        for sentence, label_sequence in zip(sentences, labels):
            for lemma, label in zip(sentence, label_sequence):
                if re.search(r'(%[1-9])', label):
                    if not lemma in lemma_to_sensekeys:
                        lemma_to_sensekeys[lemma] = [label]
                    else:
                        if not label in lemma_to_sensekeys[lemma]:
                            lemma_to_sensekeys[lemma].append(label)
        #Then save it
        with open("../../resource/Mapping_Files/lemma_to_sensekeys.txt",'w') as file:
            for elem in lemma_to_sensekeys:
                line = elem + " " + " ".join(lemma_to_sensekeys[elem])
                file.write(line + "\n")
    else:
        #Otherwise simply parse it
        with open("../../resource/Mapping_Files/lemma_to_sensekeys.txt",'r') as file:
            for line in file:
                key = line.split()[0]
                value = line.split()[1:]
                lemma_to_sensekeys[key] = value

    sensekeys_to_hypernyms = {}
    #If the following path exists, simply parse the file and extract the mapping
    if os.path.exists("../../resource/Mapping_Files/sensekeys_to_hypernyms.txt"):
        with open("../../resource/Mapping_Files/sensekeys_to_hypernyms.txt", 'r') as file:
            for line in file:
                key = line.split()[0]
                value = line.split()[1:]
                sensekeys_to_hypernyms[key] = value
    else:
        #Otherwise start the procedure to generate hypernyms
        hypernyms = extractHypernymTrees(lemma_to_sensekeys)
        sensekey_to_hypernyms = extractLowestAncestor(hypernyms)
        #Save the mapping
        with open("../../resource/Mapping_Files/sensekeys_to_hypernyms.txt", 'w') as file:
            for elem in sensekey_to_hypernyms:
                file.write(elem + ' ' + str(sensekey_to_hypernyms[elem]) + '\n')

    hypernym_labels = []
    for label_sequence in labels:
        temp = []
        for label in label_sequence:
            if re.search(r'(%[1-9])', label):
                if label in sensekeys_to_hypernyms:
                    temp.append( (sensekeys_to_hypernyms[label])[0])
                else:
                    temp.append(label)
            else:
                temp.append(label)
        hypernym_labels.append(temp)
    return hypernym_labels