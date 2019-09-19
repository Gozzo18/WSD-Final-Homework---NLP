import Utils
import os
import tensorflow as tf
import numpy as np
import Mappings

MODEL_PATH = 'Model/model.ckpt'
META_GRAPH_PATH = 'Model/model.ckpt.meta' 

OUTPUT_VOCAB_FILE = 'Mapping_Files/sensekey_output_vocabulary.txt'
LEMMAS_MAPPING = 'Mapping_Files/lemma_to_sensekeys.txt'


def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    #Parse the dataset and extract all the required informations
    test_x, token_identifier = Utils.extractTokensToPredict(input_path)
    print()

    wordnetCompression = False

    output_vocab = {}
    with open(resources_path + OUTPUT_VOCAB_FILE, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1]
            output_vocab[key] = value
    #The inverted mapping will be used later on to decode labels
    inverted_output_vocab = {v: k for k, v in output_vocab.items()}
   
    lemmas_mapping = {}
    with open(resources_path + LEMMAS_MAPPING, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1:]
            lemmas_mapping[key] = value

    #Retrieve the label identifiers
    label_identifiers = Utils.extractLabelIdentifier(test_x, token_identifier, lemmas_mapping, output_vocab, wordnetCompression)
    sequence_length, embeddings = Utils.ELMo_module(test_x, 400)
    print("Embedding size: ", np.asarray(embeddings).shape)
    print()

    #Retrieve predictions for the lemma to disambiguate
    predictions = Utils.predictions_architecture_2(embeddings, sequence_length, resources_path +  MODEL_PATH, resources_path + META_GRAPH_PATH)
    print()

    #Evaluate the labels
    labels = Utils.evaluate(predictions, token_identifier, label_identifiers)

    #Decode the labels
    decoded_labels = Utils.decodeLabels(labels, inverted_output_vocab)

    #Transform first sensekeys into wordnet synsets
    wordnet_predictions = Mappings.sensekeyToWN(decoded_labels)
    print("Retrieve BabelNet synset \n")
    #Transform wordnet synsets into babelnet synsets
    babelnet_predictions = Mappings.wordnetToBN(wordnet_predictions, resources_path + 'Mapping_Files/babelnet2wordnet.tsv')
    
    print("Writing BabelNet predictions file")
    Mappings.writePredictionFile(token_identifier, babelnet_predictions, output_path) 

def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    test_x, token_identifier = Utils.extractTokensToPredict(input_path)
    print()

    wordnetCompression = False

    output_vocab = {}
    with open(resources_path + OUTPUT_VOCAB_FILE, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1]
            output_vocab[key] = value
    #The inverted mapping will be used later on to decode labels
    inverted_output_vocab = {v: k for k, v in output_vocab.items()}
   
    lemmas_mapping = {}
    with open(resources_path + LEMMAS_MAPPING, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1:]
            lemmas_mapping[key] = value

    #Retrieve the label identifiers
    label_identifiers = Utils.extractLabelIdentifier(test_x, token_identifier, lemmas_mapping, output_vocab, wordnetCompression)
    sequence_length, embeddings = Utils.ELMo_module(test_x, 400)
    print("Embedding size: ", np.asarray(embeddings).shape)
    print()

    #Retrieve predictions for the lemma to disambiguate
    predictions = Utils.predictions_architecture_2(embeddings, sequence_length, resources_path +  MODEL_PATH, resources_path + META_GRAPH_PATH)
    print()

    #Evaluate the labels
    labels = Utils.evaluate(predictions, token_identifier, label_identifiers)

    #Decode the labels
    decoded_labels = Utils.decodeLabels(labels, inverted_output_vocab)
    
    #Transform first sensekeys into wordnet synsets
    wordnet_predictions = Mappings.sensekeyToWN(decoded_labels)
    #Transform wordnet synsets into babelnet synsets
    babelnet_predictions = Mappings.wordnetToBN(wordnet_predictions, resources_path + 'Mapping_Files/babelnet2wordnet.tsv')
    print("Retrieve WordNet Domains \n")
    domain_predictions = Mappings.bnToDomains(babelnet_predictions, resources_path + 'Mapping_Files/babelnet2wndomains.tsv')
    print("Writing Domain predictions file")
    Mappings.writePredictionFile(token_identifier, domain_predictions, output_path)  

def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    test_x, token_identifier = Utils.extractTokensToPredict(input_path)
    print()

    wordnetCompression = False

    output_vocab = {}
    with open(resources_path + OUTPUT_VOCAB_FILE, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1]
            output_vocab[key] = value
    #The inverted mapping will be used later on to decode labels
    inverted_output_vocab = {v: k for k, v in output_vocab.items()}
   
    lemmas_mapping = {}
    with open(resources_path + LEMMAS_MAPPING, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1:]
            lemmas_mapping[key] = value

    #Retrieve the label identifiers
    label_identifiers = Utils.extractLabelIdentifier(test_x, token_identifier, lemmas_mapping, output_vocab, wordnetCompression)
    sequence_length, embeddings = Utils.ELMo_module(test_x, 400)
    print("Embedding size: ", np.asarray(embeddings).shape)
    print()

    #Retrieve predictions for the lemma to disambiguate
    predictions = Utils.predictions_architecture_2(embeddings, sequence_length, resources_path +  MODEL_PATH, resources_path + META_GRAPH_PATH)
    print()

    #Evaluate the labels
    labels = Utils.evaluate(predictions, token_identifier, label_identifiers)

    #Decode the labels
    decoded_labels = Utils.decodeLabels(labels, inverted_output_vocab)
    
    #Transform first sensekeys into wordnet synsets
    wordnet_predictions = Mappings.sensekeyToWN(decoded_labels)
    #Transform wordnet synsets into babelnet synsets
    babelnet_predictions = Mappings.wordnetToBN(wordnet_predictions, resources_path + 'Mapping_Files/babelnet2wordnet.tsv')
    print("Retrieve Lexicographer prediction \n")
    lex_predictions = Mappings.bnToLexnames(babelnet_predictions, resources_path + 'Mapping_Files/babelnet2lexnames.tsv')
    print("Writing Lexicographer predictions file")
    Mappings.writePredictionFile(token_identifier, lex_predictions, output_path)   
