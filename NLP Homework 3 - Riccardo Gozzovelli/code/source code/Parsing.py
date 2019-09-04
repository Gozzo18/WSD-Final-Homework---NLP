from lxml import etree
import re

def parseDataset(training_set_file, training_gold_file):
    """
    Parse the dataset files by extracting sentences and labels. Remove punctuations and not well-formatted words where presents. 
    
    :param training_set_file: path of the training set file to parse
    :param training_gold_file: path of the training gold file to parse
    :return sentences: lists of training sentences
    :return labels: lists of training labels
    
    """

    punctuation = '!"#$&\'``()*,-./:;<=>?@[\]^_`{}~'    #List of punctuation symbols to exclude
    sentences = [] 
    labels = []
    sentence_ids = [] #Used to simplify the parsing of the label file  
    
    for event, element in etree.iterparse(training_set_file, tag="sentence"):
        current_sentence = []
        current_ids = []
        #Wait for the etree library to completely the current node
        if event == 'end':
            #For every child of the sentence tag
            for child in element:
                #Get the lemma attribute of the tag
                lemma = child.attrib['lemma']
                #Check for the presence of this particular sequence of characters associated to a wrong formatting
                if '&apos;' in lemma:
                    #If it is present, substitute it with the correct character
                    lemma = re.sub(r'(&apos;)', '\'', lemma)
                    #Check this additional case too
                    if '\'\'' in word:
                        lemma = re.sub(r'(\'\')', '\'', lemma)
                #Avoid inserting any punctuation symbol
                if not lemma in punctuation:
                    current_sentence.append(lemma)
                    #If the current tag is an instance then it is associated also to an 'id' attribute which we store
                    if child.tag == 'instance':
                        current_ids.append(child.attrib['id'])
                    #Otherwhise we add a fictional parameter
                    else:
                        current_ids.append('0')
        #Check the presence of empty sentences
        if current_sentence and current_ids:
            sentences.append(current_sentence)
            sentence_ids.append(current_ids)
        #Once all the operations are finished, clear from memory the current child
        element.clear()

    #Open gold file                    
    with open(training_gold_file, 'r') as training_gold:
        for i in range(len(sentences)):
            gold = []
            #For each sentence get the number of elements annotated lemmas
            j = len( list( filter(lambda x: not x=='0', sentence_ids[i]) ) )
            for elem in sentences[i]:
                gold.append(elem)
            #Start reading j lines from the gold file
            for line_number in range(0, j):
                line = next(training_gold)
                #Extract the id_attribute
                id_attribute = line.split()[0]
                #Retrieve the index in sentence_ids[i] associated to id_attribute
                index = sentence_ids[i].index(id_attribute)
                #Use index to update the i-th string with the label
                gold[index] = (line.split()[1]).replace('\n','')
            labels.append(gold)

    return sentences, labels