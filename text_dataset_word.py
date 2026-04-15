'''text_dataset_word.py
Functions to create, organize, and preprocess a word level text dataset
Samuel Atilano and Alex Solano
CS 443: Bio-inspired Machine Learning
Project 3: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import pickle
import pandas as pd
import tensorflow as tf
import re

from text_util import clean_review, tokenize_words


class WordLevelDataset:
    '''Object for organizing, storing, and preprecessing IMDd-like word level text datasets organized like so:

    [Document/Review] -> [Sentences of words]

    Preprocessing adopts a word-level model.
    '''
    def __init__(self, file_path='data/imdb_train.csv', min_sent_size=2, context_win_sz=2, verbose=True):
        '''WordLevelDataset constructor

        (This method is provided to you, it should not require modification)

        Parameters:
        -----------
        file_path: str.
            File path to the .csv dataset.
        min_sent_size: int.
            Don't add sentences LESS THAN this number of words to the corpus (skip over them). This is important because
            it removes empty sentences (bad parsing) and those with not enough word context.
        context_win_sz: int.
            The number of words on either side of each current target word that define the target's word context words.
        verbose: bool.
            If False, turns off all debugging print outs.

        '''
        self.file_path = file_path
        self.min_sent_size = min_sent_size
        self.context_win_sz = context_win_sz
        self.verbose = verbose

        # KEEP THE FOLLOWING PLACEHOLDERS, THESE ALL SHOULD BE SET BY YOUR METHODS
        self.reviews_raw = None  # raw review strings
        self.corpus = None  # Corpus list of lists organized: sentences across reviews/words in each sentence
        self.vocab = None  # Unique words in the corpus
        self.word2ind_map = None  # Maps word str to int codes
        self.ind2word_map = None  # Maps word int codes to strs
        self.context_words_int = None  # tf.int32 tensor. Flat list of context word int codes in each sent in corpus.
        self.target_words_int = None  # tf.int32 tensor. Flat list of target word int codes in every sent in corpus.

    def get_filepath(self):
        '''Get the filepath to the dataset .CSV data file.'''
        
        return self.file_path

    def get_reviews(self):
        '''Get the raw text reviews represented as strings.'''
        
        return self.reviews_raw

    def get_corpus(self):
        '''Get the corpus, a list of lists organized: sentences across reviews/words in each sentence'''
        
        return self.corpus

    def get_context_words(self):
        '''Get the (N,) int-coded context word tensor for context words across the corpus.'''
        
        return self.context_words_int

    def get_target_words(self):
        '''Get the (N,) int-coded target word tensor for target words across the corpus.'''
        
        return self.target_words_int

    def get_word2ind_map(self):
        '''Get dictionary that looks up a word index (int) by its string.'''
        
        return self.word2ind_map

    def get_ind2word_map(self):
        '''Get dictionary that uses a word int code to look up its word string'''
        
        return self.ind2word_map

    def get_vocab(self):
        '''Get the vocabulary, the unique list of words in the corpus'''
        
        return self.vocab

    def save_vocab(self, full_path='export/vocab.pkl'):
        '''Saves the vocabulary to disk to quick retrieval later without re-parsing the whole text dataset.

        (This method is provided to you, it should not require modification)

        Parameters:
        -----------
        full_path: str.
            File path to put the saved vocab.
        '''
        with open(full_path, 'wb') as fp:
            pickle.dump(self.vocab, fp)

    def save_corpus(self, full_path='export/corpus.pkl'):
        '''Saves the corpus to disk to quick retrieval later without re-parsing the whole text dataset.

        (This method is provided to you, it should not require modification)

        Parameters:
        -----------
        full_path: str.
            File path to put the saved corpus.
        '''
        with open(full_path, 'wb') as fp:
            pickle.dump(self.corpus, fp)

    def save_word2ind_map(self, full_path='export/word2ind_map.pkl'):
        '''Saves the word2index dictionary to disk to quick retrieval later without re-parsing the whole text dataset.

        (This method is provided to you, it should not require modification)

        Parameters:
        -----------
        full_path: str.
            File path to put the saved word2index dictionary.
        '''
        with open(full_path, 'wb') as fp:
            pickle.dump(self.word2ind_map, fp)

    def save_ind2word_map(self, full_path='export/ind2word_map.pkl'):
        '''Saves the index2word dictionary to disk to quick retrieval later without re-parsing the whole text dataset.

        (This method is provided to you, it should not require modification)

        Parameters:
        -----------
        full_path: str.
            File path to put the saved index2word dictionary.
        '''
        with open(full_path, 'wb') as fp:
            pickle.dump(self.ind2word_map, fp)

    def load(self, N_reviews):
        '''Loads the text dataset .CSV file and formats the data as a 1D Python list, where each item is a single review
        represented as a string. Example:

        [<review 1 str>, <review 2 str>, <review 3 str>, ...]

        Parameters:
        -----------
        N_reviews: int.
            Number of reviews to retrieve/return sequentially, starting from the first review.
            If the user passes in -1, retrieve ALL available reviews.

        Returns:
        --------
        Python list of str. len=N_reviews
            Retrieved reviews, with each whole review represented as a single string (see example above).
        '''
        
        #Loading dataset
        file_path = self.get_filepath()
        data_df = pd.read_csv(file_path)

        #turning review column into list of N_reviews size
        review_list = []
        if N_reviews == -1:
            review_list = data_df['review'].tolist()
        else:
            review_list = data_df['review'].head(N_reviews).tolist()
        
        #Assigning instance variable
        self.reviews_raw = review_list

        return review_list

    def make_corpus(self, reviews, min_sent_size=2):
        '''Make the text corpus of the IMDb dataset.

        Transforms text documents (list of strings) into a list of list of words (both Python lists).
        The format is [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].

        For the IMDb data, this transforms a list of reviews (each is a single string) into a list of
        sentences, where each sentence is represented as a list of string words. So the elements of the
        resulting list are the i-th sentence overall. Note that this means that we lose information about WHICH review
        the sentence comes from, which is useful, but not needed in this project.

        Parameters:
        -----------
        reviews: Python list of str. len=N_reviews
            Retrieved reviews, with each whole review represented as a single string. Formated as:
            [<review 1 str>, <review 2 str>, <review 3 str>, ...]
        min_sent_size: int.
            Don't add sentences LESS THAN this number of words to the corpus (skip over them). This is important because
            it removes empty sentences (bad parsing) and those with not enough word context.

        Returns:
        -----------
        Python list of str.
            The corpus represented as: [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].

        TODO:
        1. Load in the requested number of reviews and ratings.
        2. Split each review into sentences based on periods, question marks, OR exclamation points.
        3. Tokenize the sentence into individual word strings (via provided `tokenize_words` function in `text_util.py`)
        4. Make sure only sentences get added to the corpus that are AT LEAST as long as the min length.
        '''
        
        corpus = []
        #Grabbing individual review from reviews
        for review in reviews:
            #Splitting that review into list of sentences separated by period
            sentences = re.split(r'[.!?]', review)

            #Loop through list of sentences
            for sentence in sentences:
                #Tokenizing the sentence into a list of words
                word_strings = tokenize_words(sentence)

                #Appending sentence/list of words to corpus if size is atleast min_sent_size
                if len(word_strings) >= min_sent_size:
                    corpus.append(word_strings)

        #Assigning Instance Variable
        self.corpus = corpus

        return corpus
    


    

    def make_vocabulary(self, corpus):
        '''Define the vocabulary in the corpus (unique words). Finds and returns a list of the unique words in the
        corpus.

        Parameters:
        -----------
        corpus: Python list of lists.
            Sentences of strings (words in each sentence).

        Returns:
        -----------
        Python list of str. len=vocab_sz.
            List of unique words in the corpus.
        '''
        
        unique_words = []
        for sentence in corpus:
            for word in sentence:
                if word not in unique_words:
                    unique_words.append(word)
                else:
                    continue
        
        #Assigning Instance Variable
        self.vocab = unique_words

        return unique_words


    def make_word2ind_mapping(self, vocab):
        '''Create dictionary that looks up a word index (int) by its string.
        Indices for each word are in the range [0, vocab_sz-1].

        Parameters:
        -----------
        vocab: Python list of str.
            Unique words in corpus.

        Returns:
        -----------
        Python dictionary. key,value pairs: str,int
        '''

        word2ind_dict = {}
        int_code = 0
        for word in vocab:
            word2ind_dict[word] = int_code
            int_code += 1

        #Assigning dictionary to instance var
        self.word2ind_map = word2ind_dict

        return word2ind_dict

    def make_ind2word_mapping(self, vocab):
        '''Create dictionary that uses a word int code to look up its word string
        Indices for each word are in the range [0, vocab_sz-1].

        Parameters:
        -----------
        vocab: Python list of str.
            Unique words in corpus.

        Returns:
        -----------
        Python dictionary with key,value pairs: int,str
        '''
        
        ind2word_dict = {}
        int_code = 0
        for word in vocab:
            ind2word_dict[int_code] = word
            int_code += 1

        #Assigning dictionary to instance var
        self.ind2word_map = ind2word_dict

        return ind2word_dict

    def make_target_context_word_lists(self, corpus, word2ind, context_win_sz=2):
        '''Make the target word array ("training samples") and context word array ("classes").

        Parameters:
        -----------
        corpus: Python list of lists of str.
            List of sentences, each of which is a list of words (str).
            Format: [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...]
        word2ind: Dictionary.
            Maps word string -> int code index. Range is [0, vocab_sz-1] inclusive.
        context_win_sz: int.
            How many words to include before/after the target word in sentences for context.

        Returns:
        --------
        tf.int32 tensor. shape=(N,).
            The int-coded target words in the corpus,
        tf.int32 tensor. shape=(N,).
            The int-coded context words.

        Every pair of target and context words occupy the i-th position of the target and context word lists that this
        function builds up. This means that there will likely be chains/sequences of repeating target words (when a
        target word has multiple context words in the context window, which is usually the case).

        Example:
        --------
        corpus = [['neural', 'nets', 'are', 'fun'], ...]
        word2ind: 'neural':0, 'nets':1, 'are':2, 'fun':3
        context_win_sz=1
        Then we will have:
        target_words_int =  [0, 1, 1, 2, 2, 3]
        context_words_int = [1, 0, 2, 1, 3, 2]

        NOTE:
        - As the example above illustrates, the number of context words in the window is NOT constant because of
        sentence edge effects.
        - The length of target_words_int and context_words_int MUST be equal!
        '''
        target_words_int = []
        context_words_int = []

        #Looping through each sentence
        for sentence in corpus:
            #Looping through each word in the sentence
            for word_index in range(len(sentence)):
    
                target_word = sentence[word_index]

                #Creating context window range
                start = max(0, word_index - context_win_sz)
                end = min(len(sentence), word_index + context_win_sz + 1)

                #Looping through context and excluding the word itself
                for i in range(start, end):
                    if sentence[i] == target_word: #Looking at the target word
                        continue
                    else:
                        context_word = sentence[i]

                        #Grab word2ind ints and append to list
                        target_ind = word2ind[target_word]
                        context_ind = word2ind[context_word]

                        target_words_int.append(target_ind)
                        context_words_int.append(context_ind)
                    
        #Converting lists into tensors
        target_words_int = tf.convert_to_tensor(target_words_int, dtype=tf.int32)
        context_words_int = tf.convert_to_tensor(context_words_int, dtype=tf.int32)

        #Assigning Instance Variables
        self.target_words_int = target_words_int
        self.context_words_int = context_words_int

        return target_words_int, context_words_int

    def process(self, N_reviews=10000):
        '''Gets and preprocesses the IMDb dataset appropriately for training the Skipgram neural network.
        This is a wrapper function to automate the functions you have already written.

        Parameters:
        -----------
        N_reviews: int.
            Number of reviews to load from the file.

        Returns:
        --------
        tf.int32 tensor. shape=(N,).
            The int-coded target words in the corpus.
        tf.int32 tensor. shape=(N,).
            The int-coded context words.
        Python list of str.
            The vocabulary / list of unique words in the corpus.

        TODO:
        1. Use your existing methods to preprocess the dataset.
        2. Assign key variables (e.g. vocab, word2ind map, ...) as instance variables for quick retrieval after the
        preprocessing completes. For the constructor for a full list of fields to set.
        '''

        if self.verbose:
            print(f'Number of target words: {len(self.target_words_int)}')
            print(f'Number of context words: {len(self.context_words_int)}')

        return self.target_words_int, self.context_words_int, self.vocab
