import Parsing
import Hypernyms
import CleanAndOrder
import Mappings
import Vocabulary
import ELMo
import CreateDataset
import Utils
import BiLSTM
import Predict

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
import numpy as np
import time

"""
 NOTE! SET TRAIN VARIABLE TO TRUE OR FALSE
"""
TRAIN = False

"""
YOU CAN NOW GO ON
"""

#FILE PATHS
TRAIN_EMBEDDING_FILE = 'Embeddings/semcor_sentence_embeddings.pickle'
TRAIN_PADDED_SEQUENCES_FILE = 'Embeddings/padded_semcor_sentence_embeddings.pickle'
TRAINING_SET_FILE = 'Semcor/semcor.data.xml'
TRAINING_GOLD_FILE = 'Semcor/semcor.gold.key.txt'

DEV_EMBEDDING_FILE = 'Embeddings/semEval07_sentence_embeddings.pickle'
DEV_PADDED_SEQUENCES_FILE = 'Embeddings/padded_semEval07_sentence_embeddings.pickle'
DEV_SET_FILE = 'semeval2007/semeval2007.data.xml'
DEV_GOLD_FILE = 'semeval2007/semeval2007.gold.key.txt'

BABELNET_TO_WORDNET_FILE = 'Mapping_Files/babelnet2wordnet.tsv'
BABELNET_TO_WNDOMAINS_FILE = 'Mapping_Files/babelnet2wndomains.tsv'
BABELNET_TO_LEXNAMES_FILE =  'Mapping_Files/babelnet2lexnames.tsv'

#NETWORK HYPER-PARAMETERS
EMBEDDING_SIZE = 400
MAX_LENGTH = 60
HIDDEN_SIZE = 1024
BATCH_SIZE = 32
EPOCHS = 20

#NETWORK PATH FILES
LOGGING_DIR = 'Logging/tensorflow_model_11/'
CHECKPOINT_SAVE_FILE =  "Model/architecture1/experiment2/checkpoint/model.ckpt"
CHECKPOINT_PATH = "Model/architecture1/experiment2/checkpoint/"
META_GRAPH_PATH = 'Model/architecture1/experiment2/complete/model.ckpt.meta'
COMPLETE_MODEL_FILE = 'Model/architecture1/experiment2/complete/model.ckpt'

def train():
    #Parse training set and development set files
    train_sentences, train_labels = Parsing.parseDataset(TRAINING_SET_FILE, TRAINING_GOLD_FILE)
    dev_sentences, dev_labels = Parsing.parseDataset(DEV_SET_FILE, DEV_GOLD_FILE)
    print("Number of training sentences ", len(train_sentences))
    print("Number of development sentences ", len(dev_sentences))
    print()

    #Define the type of model to create
    hypernymsCompression = False #If true, use hypernymes compression technique
    wordnetCompression = False #If true use wordnet compression technique
    singleTaskLearning = True #If true use a multi task network

    print("You are currently working with:\nWordnet Compression = %r\tHypernyms Compression = %r\tSingle-Task Learning = %r\n" %(wordnetCompression, hypernymsCompression, singleTaskLearning))

    #Depending on the type of compression, use different labels
    if wordnetCompression:
        print("Compressing labels\n")
        #Return compressed labels
        train_labels = Mappings.lemmasToSynsets(train_sentences, train_labels, True)
        dev_labels = Mappings.lemmasToSynsets(dev_sentences, dev_labels, False)
        OUTPUT_VOCABULARY_FILE = '../../resource/Mapping_Files/wordnet_output_vocabulary.txt'
    elif hypernymsCompression:
        #Return compressed labels
        print("Compressing labels\n")
        train_hypernym_labels = Hypernyms.sensekeysToHypernyms(train_sentences, train_labels)
        dev_hypernym_labels = Hypernyms.sensekeysToHypernyms(dev_sentences, dev_labels)
        OUTPUT_VOCABULARY_FILE = '../../resource/Mapping_Files/hypernyms_output_vocabulary.txt'
    else:
        Mappings.lemmaToSensekey(train_sentences, train_labels)
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
    train_y, train_sequence_length = CreateDataset.singleTaskTrainingSet(train_sorted_labels, output_vocabulary, MAX_LENGTH)
    #Retrieve the development inputs and labels for the network 
    dev_x = CreateDataset.padDatasets(DEV_EMBEDDING_FILE, MAX_LENGTH, EMBEDDING_SIZE, DEV_PADDED_SEQUENCES_FILE)
    dev_y, dev_sequence_length = CreateDataset.singleTaskTrainingSet(dev_sorted_labels, output_vocabulary,  MAX_LENGTH)
    #Retrieve training and development domain and lexname labels in case of a multitasking architecture
    if not singleTaskLearning:
        train_domain_y, _ = CreateDataset.singleTaskTrainingSet(train_domain_labels, domain_output_vocabulary, MAX_LENGTH)
        train_lexname_y, _ = CreateDataset.singleTaskTrainingSet(train_lex_labels, lex_output_vocabulary, MAX_LENGTH)
        dev_domain_y, _ = CreateDataset.singleTaskTrainingSet(dev_domain_labels, domain_output_vocabulary, MAX_LENGTH)
        dev_lexname_y, _ = CreateDataset.singleTaskTrainingSet(dev_lex_labels, lex_output_vocabulary,  MAX_LENGTH)
    print("Dimension of train_x: ", train_x.shape)
    print("Dimension of train_y: ", train_y.shape)
    print("Dimension of dev_x: ", dev_x.shape)
    print("Dimension of dev_y: ", dev_y.shape)
    print()

    #Neural network model definition
    OUTPUT_VOCABULARY_LENGTH = len(output_vocabulary)
    if not singleTaskLearning:
        DOMAIN_VOCABULARY_LENGTH = len(domain_vocab)
        LEXNAME_VOCABULARY_LENGTH = len(lexname_vocab)

    tf.reset_default_graph()
    #Graph initialization
    g = tf.Graph()
    with g.as_default():
        if singleTaskLearning:
            print("Creating single-task learning architecture")
            inputs, labels, input_prob, output_prob, state_prob, sequence_length, loss, train_op, acc = BiLSTM.simpleBiLSTM(BATCH_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_VOCABULARY_LENGTH) 
        else:
            print("Creating multi-task learning architecture")
            inputs, sensekey_labels, domain_labels, lexname_labels, keep_prob, lambda_1, lambda_2, sequence_length, lr, sensekey_loss, domain_loss, lexname_loss, train_op, acc =  multitaskBidirectionalModel(BATCH_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, MAX_LENGTH, OUTPUT_VOCABULARY_LENGTH, DOMAIN_VOCABULARY_LENGTH, LEXNAME_VOCABULARY_LENGTH)
        saver = tf.train.Saver() 

    n_iterations = int(np.ceil(len(train_x)/BATCH_SIZE))
    n_dev_iterations = int(np.ceil(len(dev_x)/BATCH_SIZE))
   
    #MAIN TRAINING LOOP
    with tf.Session(graph=g) as sess:
        #Check for the presence of checkpoints in order to restore training
        if tf.train.latest_checkpoint(CHECKPOINT_PATH):
            print("Checkpoint present. Restoring model.")
            saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        else:
            print("Model not present. Initializing variables.")
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer()) 
        train_writer = tf.summary.FileWriter(LOGGING_DIR, sess.graph)
        print("\nStarting training...")
        #We use try-catch in order to save the model when the training is stopped through a keyboard interrupt event 
        try:
            for epoch in range(0, EPOCHS):
                if singleTaskLearning:
                    print("\nEpoch", epoch + 1)
                    epoch_loss, epoch_acc = 0., 0.
                    mb = 0
                    print("======="*10)
                    start = time.perf_counter()
                    for batch_x, batch_y, batch_seq_length, in Utils.batch_generator(train_x, train_y, train_sequence_length, BATCH_SIZE):
                        mb += 1
                        _, loss_val, acc_val = sess.run([train_op, loss, acc], feed_dict={inputs: batch_x, labels: batch_y, sequence_length:batch_seq_length, input_prob:0.5, output_prob:0.5, state_prob:1.0})
                        epoch_loss += loss_val
                        epoch_acc += acc_val
                        print("{:.2f}%\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f} ".format(100.*mb/n_iterations, epoch_loss/mb, epoch_acc/mb), end="\r")
                    elapsed = time.perf_counter() - start
                    print('Elapsed %.3f seconds.' % elapsed)
                    epoch_loss /= n_iterations
                    epoch_acc /= n_iterations
                    Utils.add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
                    Utils.add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
                    print("\n")
                    print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
                    print("======="*10)
                    # DEV EVALUATION
                    dev_loss, dev_acc = 0.0, 0.0
                    for batch_x, batch_y, batch_seq_length in Utils.batch_generator(dev_x, dev_y, dev_sequence_length, BATCH_SIZE):
                        loss_val, acc_val = sess.run([loss, acc], feed_dict={inputs: batch_x, labels: batch_y, sequence_length:batch_seq_length, input_prob:0.5, output_prob:0.5, state_prob:1.0})
                        dev_loss += loss_val
                        dev_acc += acc_val
                    dev_loss /= n_dev_iterations
                    dev_acc /= n_dev_iterations
                    Utils.add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
                    Utils.add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
                    print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}".format(dev_loss, dev_acc))
                    #Save checkpoints every two epochs
                    if epoch%2 == 0:
                        save_path = saver.save(sess, CHECKPOINT_SAVE_FILE)
                else:
                    print("\nEpoch", epoch + 1)
                    epoch_sensekey_loss, epoch_domain_loss, epoch_lexname_loss, epoch_acc, epoch_f1 = 0., 0., 0., 0., 0.
                    mb = 0
                    print("======="*10)
                    start = time.perf_counter()
                    for batch_x, batch_y, batch_domain_y, batch_lexname_y, batch_seq_length, in alternative_batch_generator(train_x, train_y, train_domain_y, train_lexname_y, train_sequence_length, BATCH_SIZE):
                        mb += 1
                        _, sensekey_loss_val, domain_loss_val, lexname_loss_val, acc_val = sess.run([train_op, sensekey_loss, domain_loss, lexname_loss, acc], feed_dict={sensekey_labels: batch_y, domain_labels: batch_domain_y, lexname_labels:batch_lexname_y, lambda_1:1.0, lambda_2:1.0, keep_prob: 0.8, inputs: batch_x, sequence_length:batch_seq_length, lr:learning_rate})                     
                        epoch_sensekey_loss += sensekey_loss_val
                        epoch_domain_loss += domain_loss_val
                        epoch_lexname_loss += lexname_loss_val
                        epoch_acc += acc_val
                        print("{:.2f}%\tSensekey Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(100.*mb/n_iterations, epoch_sensekey_loss/mb, epoch_acc/mb), end="\r")
                    elapsed = time.perf_counter() - start
                    print('Elapsed %.3f seconds.' % elapsed)
                    print("{:.2f}%\tSensekey Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(100.*mb/n_iterations, epoch_sensekey_loss/mb, epoch_acc/mb), end="\r")
                    epoch_sensekey_loss /= n_iterations
                    epoch_domain_loss /= n_iterations
                    epoch_lexname_loss /= n_iterations
                    epoch_acc /= n_iterations
                    Utils.add_summary(train_writer, "epoch_sensekey_loss", epoch_sensekey_loss, epoch)
                    Utils.add_summary(train_writer, "epoch_domain_loss", epoch_domain_loss, epoch)
                    Utils.add_summary(train_writer, "epoch_lexname_loss", epoch_lexname_loss, epoch)
                    Utils.add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
                    print("\n")
                    print()
                    print("Train Sensekey Loss: {:.4f}".format(epoch_sensekey_loss))
                    print("Train Domain Loss: {:.4f}".format(epoch_domain_loss))
                    print("Train Lexname Loss: {:.4f}".format(epoch_lexname_loss))
                    print("======="*10)
                    # DEV EVALUATION
                    dev_loss, dev_acc, dev_f1 = 0.0, 0.0, 0.0
                    for batch_x, batch_y, batch_domain_y, batch_lexname_y, batch_seq_length in alternative_batch_generator(dev_x, dev_y, dev_domain_y, dev_lexname_y, dev_sequence_length, BATCH_SIZE):
                        loss_val, acc_val = sess.run([sensekey_loss, acc], feed_dict={sensekey_labels: batch_y, domain_labels: batch_domain_y, lexname_labels: batch_lexname_y, lambda_1: 1.0, lambda_2: 1.0, keep_prob: 0.8, inputs: batch_x, sequence_length:batch_seq_length, lr:learning_rate})
                        dev_loss += loss_val
                        dev_acc += acc_val
                    dev_loss /= n_dev_iterations
                    dev_acc /= n_dev_iterations
                    Utils.add_summary(train_writer, "epoch_val_loss", dev_loss, epoch)
                    Utils.add_summary(train_writer, "epoch_val_acc", dev_acc, epoch)
                    print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}".format(dev_loss, dev_acc))
                    print()
                    if epoch%2 == 0:
                        save_path = saver.save(sess, CHECKPOINT_SAVE_FILE)
        except KeyboardInterrupt:
            print("Keyboard interruption. Saving")
            save_path = saver.save(sess, COMPLETE_MODEL_FILE)
            train_writer.close()
        
        save_path = saver.save(sess, COMPLETE_MODEL_FILE)
        train_writer.close()


def predict():

    #Parse the dataset
    sentences, ids = Predict.extractTokensToPredict(DEV_SET_FILE)

    #Retrieve the lemma => sensekey mapping
    if wordnetCompression:
        #Retrieve the word/synset => int mapping
        vocab_file = 'Mapping_Files/wordnetSynsets_output_vocabulary.txt'
        #Retrieve the lemma => wordnet synset mapping
        intermediate_file = 'Mapping_Files/lemma_to_wn.txt'
    elif hypernymsCompression:
        #Retrieve the word/hypernym => int mapping
        vocab_file = 'Mapping_Files/.txt'
        #Retrieve the lemma => hypernym mapping
        intermediate_file = 'Mapping_Files/lemma_to_hypernyms.txt'
    else:
        #Retrieve the word/sensekey => int mapping
        vocab_file = 'Mapping_Files/sensekey_output_vocabulary.txt'
        #Retrieve the lemma => sensekey mapping
        intermediate_file = 'Mapping_Files/lemma_to_sensekeys.txt'

    output_vocab = {}
    with open(vocab_file, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1]
            output_vocab[key] = value
    #The inverted mapping will be used later on to decode labels
    inverted_output_vocab = {v: k for k, v in output_vocab.items()}
   
    lemmas_mapping = {}
    with open(intermediate_file, 'r') as file:
        for line in file:
            key = line.split()[0]
            value = line.split()[1:]
            lemmas_mapping[key] = value

    #Retrieve the label identifiers
    label_identifiers = Predict.extractLabelIdentifier(sentences, ids, lemmas_mapping, output_vocab, wordnetCompression)
    sequence_length, embeddings = Predict.ELMo_module(sentences, 400)
    print("Embedding size: ", embeddings.shape)
    print()

    #Retrieve predictions for the lemma to disambiguate
    predictions = Predict.predictions_architecture_1_2(embeddings, sequence_length, COMPLETE_MODEL_FILE, META_GRAPH_PATH)
    print()

    #Evaluate the labels
    labels = Predict.evaluate(predictions, ids, label_identifiers)

    #Decode the labels
    decoded_labels = Predict.decodeLabels(labels, inverted_output_vocab)

    Predict.F1Score(decoded_labels, DEV_GOLD_FILE, wordnetCompression)

if __name__ == "__main__":
    
    if TRAIN:
        train()
    else:
        wordnetCompression = False
        hypernymsCompression = False
        predict()