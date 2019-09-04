import Parsing
import WNsynsets
import Hypernyms
import CleanAndOrder
import Mappings
import Vocabulary
import ELMo
import CreateDataset


#FILE PATHS
TRAIN_EMBEDDING_FILE = '../../resource/Embeddings/semcor_sentence_embeddings.pickle'
TRAIN_PADDED_SEQUENCES_FILE = '../../resource/Embeddings/padded_semcor_sentence_embeddings.pickle'
TRAINING_SET_FILE = '../../resource/Semcor/semcor.data.xml'
TRAINING_GOLD_FILE = '../../resource/Semcor/semcor.gold.key.txt'

DEV_EMBEDDING_FILE = '../../resource/Embeddings/semEval07_sentence_embeddings.pickle'
DEV_PADDED_SEQUENCES_FILE = '../../resource/Embeddings/padded_semEval07_sentence_embeddings.pickle'
DEV_SET_FILE = '../../resource/semeval2007/semeval2007.data.xml'
DEV_GOLD_FILE = '../../resource/semeval2007/semeval2007.gold.key.txt'

BABELNET_TO_WORDNET_FILE = '../../resource/Mapping_Files/babelnet2wordnet.tsv'
BABELNET_TO_WNDOMAINS_FILE = '../../resource/Mapping_Files/babelnet2wndomains.tsv'
BABELNET_TO_LEXNAMES_FILE =  '../../resource/Mapping_Files/babelnet2lexnames.tsv'

#NETWORK HYPER-PARAMETERS
EMBEDDING_SIZE = 400
MAX_LENGTH = 60



if __name__ == "__main__":
    
    #Parse training set and development set files
    train_sentences, train_labels = Parsing.parseDataset(TRAINING_SET_FILE, TRAINING_GOLD_FILE)
    dev_sentences, dev_labels = Parsing.parseDataset(DEV_SET_FILE, DEV_GOLD_FILE)
    print("Number of training sentences ", len(train_sentences))
    print("Number of development sentences ", len(dev_sentences))
    print()

    #Define the type of model to create
    hypernymsCompression = True #If true, use hypernymes compression technique
    wordnetCompression = False #If true use wordnet compression technique
    singleTaskLearning = True #If true use a multi task network
    isSimple = True # If false use attention layer, else use simple Bi-LSTM

    print("You are currently working with:\nWordnet Compression = %r\tHypernyms Compression = %r\tMulti-Task Learning = %r\n" %(wordnetCompression, hypernymsCompression, singleTaskLearning))

    #Depending on the type of compression, use different labels
    if wordnetCompression:
        print("Compressing labels\n")
        #Return compressed labels
        _, train_labels = WNsynsets.lemmasToSynsets(train_sentences, train_labels, True, wordnetCompression)
        _, dev_labels = WNsynsets.lemmasToSynsets(dev_sentences, dev_labels, False, wordnetCompression)
        OUTPUT_VOCABULARY_FILE = '../../resource/Mapping_Files/wordnet_output_vocabulary.txt'
    elif hypernymsCompression:
        #Return compressed labels
        print("Compressing labels\n")
        train_hypernym_labels = Hypernyms.sensekeysToHypernyms(train_sentences, train_labels)
        dev_hypernym_labels = Hypernyms.sensekeysToHypernyms(dev_sentences, dev_labels)
        OUTPUT_VOCABULARY_FILE = '../../resource/Mapping_Files/hypernyms_output_vocabulary.txt'
    else:
        OUTPUT_VOCABULARY_FILE = '../../resource/Mapping_Files/sensekey_output_vocabulary.txt'

    #Clear and order training set
    print("Filter training set by removing useless sentences and order it")
    if hypernymsCompression:
        filtered_train_sentences, filtered_train_labels = CleanAndOrder.filterList(train_sentences, train_hypernym_labels)
    else:
        filtered_train_sentences, filtered_train_labels = CleanAndOrder.filterList(train_sentences, train_labels)
    print("Number of filtered training sentences ", len(filtered_train_sentences))
    print()
    train_sorted_sentences, train_sorted_labels, train_length_group = CleanAndOrder.sortAndGroup(filtered_train_sentences, filtered_train_labels, True)
    dev_sorted_sentences, dev_sorted_labels, dev_length_group = CleanAndOrder.sortAndGroup(dev_sentences, dev_labels, False)

    print("Retrieving mappings between WordNet synsets => BabelNet synsets, BabelNet synsets => WordNet Domains and BabelNet synsets => Lexical Names")
    mapping_file_list = [BABELNET_TO_WORDNET_FILE, BABELNET_TO_WNDOMAINS_FILE, BABELNET_TO_LEXNAMES_FILE]
    wordNet_to_babelNet, babelNet_to_wnDomain, babelNet_to_lexNames = Mappings.extractMappings(mapping_file_list)
    print("WordNet => BabelNet mapping length: ", len(wordNet_to_babelNet))
    print("BabelNet => Domain mapping length ", len(babelNet_to_wnDomain))
    print("BabelNet => Lexnames mapping length ", len(babelNet_to_lexNames))
    print()

    #Define the output vocabulary in order to map labels from string to integers
    print("Retrieving output vocabulary")
    output_vocabulary = Vocabulary.extractOutputVocabulary(train_sorted_labels, dev_sorted_labels, OUTPUT_VOCABULARY_FILE)
    print("Size of output_vocabulary: %i\n" %len(output_vocabulary))
    if singleTaskLearning:
        print("Retrieving Babelnet, Domain and Lexname labels and vocabularies")
        train_bn_labels = Mappings.wnToBn(train_sorted_labels, wordNet_to_babelNet, wordnetCompression)
        dev_bn_labels = Mappings.wnToBn(dev_sorted_labels, wordNet_to_babelNet, wordnetCompression)

        train_domain_labels = Mappings.bnToWnDomain(train_bn_labels, babelNet_to_wnDomain)
        dev_domain_labels = Mappings.bnToWnDomain(dev_bn_labels, babelNet_to_wnDomain)

        train_lex_labels = Mappings.bnToWnLex(train_bn_labels, babelNet_to_lexNames)
        dev_lex_labels = Mappings.bnToWnLex(dev_bn_labels, babelNet_to_lexNames)

        bn_output_vocabulary, domain_output_vocabulary, lex_output_vocabulary = Vocabulary.multiTaskingVocabularies(train_bn_labels, train_domain_labels, train_lex_labels, dev_bn_labels, dev_domain_labels, dev_lex_labels)
        print("Size of Babelnet output_vocabulary: %i" %len(bn_output_vocabulary))
        print("Size of Domain output_vocabulary: %i" %len(domain_output_vocabulary))
        print("Size of Lexname output_vocabulary: %i" %len(lex_output_vocabulary))
        print()

    #Create the embeddings for the datasets
    ELMo.Module(TRAIN_EMBEDDING_FILE, train_length_group)
    print()
    ELMo.Module(DEV_EMBEDDING_FILE, dev_length_group)
    print()

    #Retrieve the training inputs and labels for the network 
    train_x = CreateDataset.padDatasets(TRAIN_EMBEDDING_FILE, MAX_LENGTH, EMBEDDING_SIZE, TRAIN_PADDED_SEQUENCES_FILE)
    train_y, train_sequence_length = CreateDataset.singleTaskTrainingSet(train_sorted_labels, output_vocabulary, MAX_LENGTH, isSimple)
    #Retrieve the development inputs and labels for the network 
    dev_x = CreateDataset.padDatasets(DEV_EMBEDDING_FILE, MAX_LENGTH, EMBEDDING_SIZE, DEV_PADDED_SEQUENCES_FILE)
    dev_y, dev_sequence_length = CreateDataset.singleTaskTrainingSet(dev_sorted_labels, output_vocabulary,  MAX_LENGTH, isSimple)
    #Retrieve training and development domain and lexname labels in case of a multitasking architecture
    if not singleTaskLearning:
        train_domain_y, _ = CreateDataset.singleTaskTrainingSet(train_domain_labels, domain_output_vocabulary, MAX_LENGTH, False)
        train_lexname_y, _ = CreateDataset.singleTaskTrainingSet(train_lex_labels, lex_output_vocabulary, MAX_LENGTH, False)
        dev_domain_y, _ = CreateDataset.singleTaskTrainingSet(dev_domain_labels, domain_output_vocabulary, MAX_LENGTH, False)
        dev_lexname_y, _ = CreateDataset.singleTaskTrainingSet(dev_lex_labels, lex_output_vocabulary,  MAX_LENGTH, False)
    print("Dimension of train_x: ", train_x.shape)
    print("Dimension of train_y: ", train_y.shape)
    print("Dimension of dev_x: ", dev_x.shape)
    print("Dimension of dev_y: ", dev_y.shape)
    print()

