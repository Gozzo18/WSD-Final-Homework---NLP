def filterList(sentences, labels):
    """
    Filter the two lists by removing repeated elements. The filtering is done by adding each label sequence to a set.
    Therefore every repeated sequence is not added.
    
    :param sentences: list of sentences to filter
    :param labels: list of labels to filer
    :return filtered_sentences: list of sentences without repeted sequences
    :return filtered_labels: list of labels without repeted sequences
    
    """
    
    seen_label = set() 
    filtered_sentences = []
    filtered_labels = []
    identifier = 1
    
    for words, labels in zip(sentences, labels): 
        #Reconstruct the label sequence
        sequence_labels = ' '.join(labels)
        #Check its presence in the set
        if sequence_labels not in seen_label:
            seen_label.add(sequence_labels)
            filtered_labels.append(labels)
            #Add an identifier to the list of sentences, useful for further purposes
            words.append(str(identifier))
            filtered_sentences.append(words)
            identifier += 1
        else:
            #In case a sequence with a sensekey is duplicated we don't discard it
            if '::' in sequence_labels:
                filtered_labels.append(labels)
                words.append(str(identifier))
                filtered_sentences.append(words)
                identifier += 1
                
    return filtered_sentences, filtered_labels


def sortAndGroup(sentences, labels, isTrain):
    
    """
    Sort the sentence and label lista by the length attribute of each element 
    
    :param sentences: list of splitted sentences
    :param labels: list of splitted labels
    :param isTrain: boolean value used to distinguish between train and dev set
    :return sorted_sentences: ordered list of splitted sentences
    :return sorted_labels: ordered list of splitted labels
    
    """

    sorted_sentences = sorted(sentences, key=len)
    if labels:
        sorted_labels = sorted(labels, key=len)
        
    #Remove identifier from sorted list
    if isTrain:
        for i in range(len(sorted_sentences)):
            sorted_sentences[i] = sorted_sentences[i][:-1]

    grouped_sentences_vocab={}
    for sentence in sorted_sentences:
        grouped_sentences_vocab.setdefault(len(sentence), []).append(sentence)

    grouped_sentences_by_length = [grouped_sentences_vocab[n] for n in sorted(grouped_sentences_vocab)]
       
    return sorted_sentences, sorted_labels, grouped_sentences_by_length