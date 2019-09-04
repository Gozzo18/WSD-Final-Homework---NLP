import os

def extractOutputVocabulary(train_labels, dev_labels, output_vocabulary_file):
    """
    Retrieve dictionaries from training and development set. 
    
    :params train_labels: list of tokenized training labels
    :params dev_labels: list of tokenized dev labels
    
    :return output_vocabulary: dictionary containing the mapping Lemma/Sensekey => integer
    """
    
    output_vocabulary = {} 

    mapping_integer = 1
    output_vocabulary['PAD'] = 0
    #Training set
    for label_sequence in train_labels:
        for label in label_sequence:
            if not label in output_vocabulary:
                output_vocabulary[label] = mapping_integer
                mapping_integer += 1
    #Dev set
    for label_sequence in dev_labels:
        for label in label_sequence:
            if not label in output_vocabulary:
                output_vocabulary[label] = mapping_integer
                mapping_integer += 1
    #Save the output vocabulary 
    if not os.path.exists(output_vocabulary_file):
        with open(output_vocabulary_file, 'w') as file:
            for elem in output_vocabulary:
                file.write(elem + ' ' + str(output_vocabulary[elem]) + '\n')
     
    return output_vocabulary

def multiTaskingVocabularies(train_bn, train_domains, train_lexnames, dev_bn, dev_domains, dev_lexnames):
    """
    :params train_bn: list of tokenized training babelnet labels
    :params train_domains: list of tokenized training domain labels
    :params train_lexnames: list of tokenized training lexname labels
    :params dev_labels: list of tokenized dev labels 
    :params dev_bn: list of tokenized dev babelnet labels
    :params dev_domains: list of tokenized dev domain labels
    :params dev_lexnames: list of tokenized dev lexname labels
    
    :return bn_vocabulary: dictionary containing the mapping Lemma/BabelNet synset => integer
    :return domain_vocabulary: dictionary containing the mapping Lemma/Domain => integer
    :return lex_vocabulary: dictionary containing the mapping Lemma/Lexical name => integer

    """

    #BabelNet synsets vocabulary
    bn_vocabulary = {}
    bn_integer = 1
    bn_vocabulary['PAD'] = 0
    #Training set
    for bn_synsets in train_bn:
        for synset in bn_synsets:
            if not synset in bn_vocabulary:
                bn_vocabulary[synset] = bn_integer
                bn_integer += 1
    #Dev set
    for bn_synsets in dev_bn:
        for synset in bn_synsets:
            if not synset in bn_vocabulary:
                bn_vocabulary[synset] = bn_integer
                bn_integer += 1
    
    #Wordnet domains vocabulary
    domain_vocabulary = {}
    domain_integer = 1
    domain_vocabulary['PAD'] = 0
    #Training set
    for domain in train_domains:
        for elem in domain:
            if not elem in domain_vocabulary:
                domain_vocabulary[elem] = domain_integer
                domain_integer += 1
    #Dev set
    for domain in dev_domains:
        for elem in domain:
            if not elem in domain_vocabulary:
                domain_vocabulary[elem] = domain_integer
                domain_integer += 1
                
    #LEXNAMES VOCAB
    lex_vocabulary = {}
    lex_integer = 1
    lex_vocabulary['PAD'] = 0
    #Training set
    for lexname in train_lexnames:
        for elem in lexname:
            if not elem in lex_vocabulary:
                lex_vocabulary[elem] = lex_integer
                lex_integer += 1
    #Dev set
    for lexname in dev_lexnames:
        for elem in lexname:
            if not elem in lex_vocabulary:
                lex_vocabulary[elem] = lex_integer
                lex_integer += 1

    return bn_vocabulary, domain_vocabulary, lex_vocabulary

