
# coding: utf-8
# In[1]:
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

import pandas as pd
import numpy as np
#from sklearn_pandas import DataFrameMapper, cross_val_score
#import sklearn.preprocessing, sklearn.decomposition,sklearn.linear_model, sklearn.pipeline, sklearn.metrics
#from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from nltk.corpus import sentiwordnet as swn
import matplotlib
from pylab import *
import csv
import sys
import sklearn
csv.field_size_limit(sys.maxsize)
# import re
import string
############################################################-----------------------------------
#import os
import re
import gensim
import logging
import pandas as pd
import numpy as np
import nltk.data
import sys
# import sklearn
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from gensim.models import word2vec
from sklearn.preprocessing import Imputer
from sklearn import linear_model, naive_bayes, svm, preprocessing
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD,NMF ,LatentDirichletAllocation

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
from sklearn.feature_selection import SelectKBest, chi2,SelectFromModel
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np
#from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, make_scorer
import pickle

#import requests
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from gensim import models


from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.decomposition import LatentDirichletAllocation

# from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
# from keras.preprocessing import sequence, text
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers.recurrent import LSTM, GRU
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.embeddings import Embedding
# from keras.layers.normalization import BatchNormalization
# from keras.utils import np_utils
# import xgboost as xgb
##################### Function Definition #####################


def clean_document(document, remove_stopwords = True, output_format = "string"):
    """
    Input:
            document: raw text of a document
            remove_stopwords: a boolean variable to indicate whether to remove stop words
            output_format: if "string", return a cleaned string
                           if "list", a list of words extracted from cleaned string.
    Output:
            Cleaned string or list.
    """

    # Remove HTML markup
    #text = BeautifulSoup(document)

    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", document)
    #text = re.sub("[^a-zA-Z]", " ", text.get_text())

    # Split words and store to list
    text = text.lower().split()

    if remove_stopwords:

        # Use set as it has O(1) lookup time
        stops = set(stopwords.words("english"))
        words = [w for w in text if w not in stops]

    else:
        words = text

    # Return a cleaned string or list
    if output_format == "string":
        return " ".join(words)

    elif output_format == "list":
        return words


def document_to_doublelist(document, tokenizer, remove_stopwords = False):
    """
    Function which generates a list of lists of words from a document for word2vec uses.

    Input:
        document: raw text of a document
        tokenizer: tokenizer for sentence parsing
                   nltk.data.load('tokenizers/punkt/english.pickle')
        remove_stopwords: a boolean variable to indicate whether to remove stop words

    Output:
        A list of lists.
        The outer list consists of all sentences in a document.
        The inner list consists of all words in a sentence.
    """

    # Create a list of sentences
    raw_sentences = tokenizer.tokenize(document.strip())
    sentence_list = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentence_list.append(clean_document(raw_sentence, False, "list"))
    return sentence_list


def document_to_vec(words, model, num_features):
    """
    Function which generates a feature vector for the given document.

    Input:
        words: a list of words extracted from a document
        model: trained word2vec model
        num_features: dimension of word2vec vectors

    Output:
        a numpy array representing the document
    """

    feature_vec = np.zeros((num_features), dtype="float32")
    word_count = 0

    # index2word is a list consisting of all words in the vocabulary
    # Convert list to set for speed
    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            word_count += 1
            feature_vec += model[word]

    feature_vec /= word_count
    return feature_vec


def gen_document_vecs(documents, model, num_features):
    """
    Function which generates a m-by-n numpy array from all documents,
    where m is len(documents), and n is num_feature

    Input:
            documents: a list of lists.
                     Inner lists are words from each content.
                     Outer lists consist of all documents
            model: trained word2vec model
            num_feature: dimension of word2vec vectors
    Output: m-by-n numpy array, where m is len(content) and n is num_feature
    """

    curr_index = 0
    doc_feature_vecs = np.zeros((len(documents), num_features), dtype="float32")
    print "doc_feature_vecs.shape---->",doc_feature_vecs.shape
    for review in documents:
        if curr_index%1000 == 0.:
            print ("Vectorizing content %d of %d", curr_index, len(documents))
        doc_feature_vecs[curr_index] = document_to_vec(review, model, num_features)
        curr_index += 1

    return doc_feature_vecs

def add_numberical_features(train_data,train_vec):
    # function for adding numerical values to train vector
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    train_data_numerical = train_data.select_dtypes(include=numerics)
    numerical_col_names = list(train_data_numerical.columns.values)
    print len(numerical_col_names)
    print numerical_col_names
    for col_name in numerical_col_names:
        col_name_values = train_data[[str(col_name)]].values
        train_vec = sparse.hstack((train_vec, col_name_values.astype(float))).tocsr()

    print ("train_vec.shape after adding numerical values: ",train_vec.shape)


    return train_vec,numerical_col_names


def draw_learning_cuve(fitted_model,train_vec,train_data_Annotation):
    # train_sizes, train_scores, test_scores = learning_curve(lg, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
    train_sizes, train_scores, test_scores = learning_curve(fitted_model, train_vec, train_data_Annotation, n_jobs=-1, cv=10, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    plt.title("RandomForestClassifier")
    plt.legend(loc="best")
    plt.xlabel("Training Instances")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    # box-like grid
    plt.grid()

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Cross-validation score")

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1,1.1)

    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')


    #plt.show()
    fig.savefig('learn.png')
    print "your plot has been saved!!!"

def show_feature_function(fitted_model,train_vec):
    model = SelectFromModel(fitted_model, prefit=True)
    train_vec_new = model.transform(train_vec)
    # print(train_vec_new.columns[model.get_support()])
    print(train_vec_new.shape)
    print(model.get_support())




    # print type (train_vec)
    feature_idx = model.get_support()
    print len(feature_idx)
    #feature_name = train_vec.columns[feature_idx]

    # print feature_name
    # model = SelectFromModel(clf, prefit=True)
    # feature_idx = model.get_support()
    # feature_name = df.columns[feature_idx]


def feature_visualization(classifier, feature_names, top_features=50):
    '''
    This is the function which visulize both numerical and textual coef
    for each class of positive and negative
    Only for SVM
    '''
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    fig2 =plt.figure(figsize=(15, 8))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    # print top_coefficients
    # print feature_names
    # sys.exit()
    print (feature_names[1:50])
    print len(feature_names)
    print top_coefficients
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=90, ha="right",  fontsize = 8)
    #plt.show()
    fig2.savefig('feature_Coef.png')



def feature_importance_with_forrest(rfc, train_vec):


    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    #indices= indices[:20]

    # indices = np.argsort(importances)[:20]
    # print indices
    # Print the feature ranking
    print("Feature ranking:")

    #for f in range(train_vec.shape[1]):
    for f in range(500):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


    selected = importances[indices]
    selected = selected[:20]
    print len(selected)
    print range (20)
    # Plot the feature importances of the forest
    fig = plt.figure()
    plt.title("Feature importances")
    selected_indices  = indices
    selected_indices = selected_indices[:20]
    print indices
    print selected_indices
    #sys.exit()
    # plt.bar(range(train_vec.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    #plt.bar(range (20), importances[random search for parameter tuningindices], color="r", yerr=std[selected_indices], align="center")
    plt.bar(range (20), selected, color="r", align="center")
    # plt.bar(range (20), selected, color="r", align="center")
    #plt.xticks(range(train_vec.shape[1]), indices)
    plt.xticks(range(20), selected_indices)
    # plt.xlim([-1, train_vec.shape[1]])
    plt.xlim([-1, 20])
    #plt.show()
    fig.savefig('feature_Random_forrest.png')


def print_topics(model, vectorizer, top_n=20):
    ''' This is the funciton for printing topic models'''


    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

def create_txt_from_bin(model_name):
    print model_name
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True,unicode_errors='ignore')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)
    model_name_txt = model_name + "_binary_false" +".txt"
    print "model_name_txt---->",model_name_txt
    model.save_word2vec_format(model_name_txt, binary=False)
    print "model_saved"
    return model_name_txt

def load_word2vec_dic(model_name):
    '''
        A funcion for converting an embedding and return a dictionary
    '''

    print "Loading word2vec into dictionary"
    print "*******************************"
    embeddings_index = {}

    #we have to run it only once to save the model as txt file, Next time we only provide the txt location
    #model_name_txt = create_txt_from_bin(model_name)
    #sys.exit()
    model_name_txt = "/mnt/volume/data/embeddings/word2vec_twitter_model.bin_binary_false.txt"
    f = open(model_name_txt)
    #print f
    for line in tqdm(f):
        #print "line--->",line
        #sys.exit()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print "----------------"
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def sent2vec(s,embeddings_index):
    '''
    This function creates a normalized vector for the whole sentence
    '''
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())



def check_pos_tag(x, flag):
    '''
    Function to check and get the part of speech tag count of a words in a given sentence
    '''
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

def add_nlp_feature(train_data):
    #Word Count of the documents
    #Character Count of the documents
    #Average Word Density of the documents
    #Puncutation Count in the Complete Essay
    #Upper Case Count in the Complete Essay
    #Frequency distribution of Part of Speech Tags:
    train_data['char_count']  = train_data['merged_tweets_description'].apply(len)
    train_data['char_count'] = train_data['merged_tweets_description'].apply(len)
    train_data['word_count'] = train_data['merged_tweets_description'].apply(lambda x: len(x.split()))
    #train_data['word_density'] = train_data['merged_tweets_description'] / (train_data['word_count']+1)
    train_data['punctuation_count'] = train_data['merged_tweets_description'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    train_data['title_word_count'] = train_data['merged_tweets_description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    train_data['upper_case_word_count'] = train_data['merged_tweets_description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
    }

    train_data['noun_count'] = train_data['merged_tweets_description'].apply(lambda x: check_pos_tag(x, 'noun'))
    train_data['verb_count'] = train_data['merged_tweets_description'].apply(lambda x: check_pos_tag(x, 'verb'))
    train_data['adj_count'] = train_data['merged_tweets_description'].apply(lambda x: check_pos_tag(x, 'adj'))
    train_data['adv_count'] = train_data['merged_tweets_description'].apply(lambda x: check_pos_tag(x, 'adv'))
    train_data['pron_count'] = train_data['merged_tweets_description'].apply(lambda x: check_pos_tag(x, 'pron'))
    return train_data


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    #documents = []
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)
        #return documents


def add_topical_feature(train_list,train_vec,topical_type):
    print "Vectorizing..."
    count_vec = CountVectorizer(analyzer="word", min_df=5, max_df=0.8, binary=(vector_type == "Binary"),
                                ngram_range=(1, 3), lowercase=True)
    train_vec_for_topics = count_vec.fit_transform(train_list)
    print "train_vec shape...", train_vec_for_topics.shape
    print  "current vector_type: ", vector_type
    if topical_type == "LDA":
        print "LDA will start"
        # train a LDA Model
        lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, learning_method='online', max_iter=50)
        X_topics = lda_model.fit_transform(train_vec_for_topics)
        print X_topics.shape
        # show the first element
        print X_topics[0]

        print("LDA Model:")
        print_topics(lda_model, count_vec)
        print("=" * 20)

        # for col_name in X_topics:
        #   col_name_values = train_data[[str(col_name)]].values
        #  train_vec = sparse.hstack((train_vec, col_name_values.astype(float))).tocsr()

        train_vec = sparse.hstack((train_vec, X_topics.astype(float))).tocsr()
        print("train_vec.shape after topics: ", train_vec.shape)

        # train_vec = sparse.hstack((train_vec, col_name_values.astype(float))).tocsr()

        # print("train_vec.shape after adding numerical values: ", train_vec.shape)

        # train_vec = X_topics

    elif topical_type == "NMF":
        # Build a Non-Negative Matrix Factorization Model
        nmf_model = NMF(n_components=NUM_TOPICS)
        nmf_Z = nmf_model.fit_transform(train_vec)
        print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

        print("NMF Model:")
        print_topics(nmf_model, count_vec)
        print("=" * 20)
        train_vec = nmf_Z
    elif topical_type == "LSI":
        # Build a Latent Semantic Indexing Model
        lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
        lsi_Z = lsi_model.fit_transform(train_vec)
        print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
        print("LSI Model:")
        print_topics(lsi_model, count_vec)
        print("=" * 20)
        train_vec = lsi_Z

    return train_vec
#vector_type = vector_type_old
##################### End of Function Definition #####################

##################### Initialization #####################

# term_vector_type = {"TFIDF", "Binary", "Int", "Word2vec", "Word2vec_pretrained","LDA", "NMF", "LSI","Word2vec2"}
# {"TFIDF", "Int", "Binary"}: Bag-of-words model with {tf-idf, word counts, presence/absence} representation
# {"Word2vec", "Word2vec_pretrained"}: Google word2vec representation {without, with} pre-trained models
# Specify model_name if there's a pre-trained model to be loaded
# for Topic modeling reduction  topic_modeling_based_reduction = True, vector_type = LDA

# for word2vec  sent2vec_flag = True, vector_type = word2vec2, train_embedding_flag = True if you want to train on your own data,
#embedding_model_checking = True means we want to see the quality of our trained embedding models
sent2vec_flag = False
train_embedding_flag = False
embedding_model_checking = True

vector_type = "TFIDF"
# vector_type = "word2vec2"
#number of topics for topic modeling based dimension reduction
NUM_TOPICS = 20

topic_modeling_based_reduction = False # if True it creates vectors
topic_base_addition = True # in case you want to add topic feature to other feature
topical_type = "LDA"  #{LDA,NMF,LSI}


# model_name = "/home/ubuntu/embedding/GoogleNews-vectors-negative300.bin"
# model_name = "/home/ubuntu/embedding/GoogleNews-vectors-negative300.bin"
# model_name = "/home/ubuntu/embedding/glove/glove.twitter.27B.100d.txt"
model_name = "/mnt/volume/data/embeddings/word2vec_twitter_model.bin"
# model_name = "/mnt/volume/data/embeddings/word2vec_twitter_model.bin.wv.vectors.npy"
# word2vec_twitter_model.bin.wv.vectors.npy

learning_curve_flag = True

add_nlp_feature_flag = False


# model_type = {"bin", "reg"}
# Specify whether pre-trained word2vec model is binary
model_type = "bin"

# Parameters for word2vec
# num_features need to be identical with the pre-trained model
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count to be included for training
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# training_model = {"RF", "NB", "LR", "SVM", "BT", "no"}
training_model = "LR"

#show feature Importance
show_feature = True

# feature scaling = {"standard", "signed", "unsigned", "no"}
# Note: Scaling is needed for SVM
scaling = "no"

# dimension reduction = {"SVD", "chi2", "no"}
# Note: For NB models, we cannot perform truncated SVD as it will make input negative
# chi2 is the feature selectioin based on chi2 independence test
dim_reduce = "chi2"
num_dim = 50

# campaign, and target entity
#campaign_name = "election2016"
#target_name = "hillary_clinton_sentiment_counts_pos_negs_hashtags_pos_neg_new"
# train data file
#train_data_file = "../data/" + campaign_name + "/" + target_name + ".csv"
# model file
save_model = False
# model_file = "../models/" + campaign_name + "/" + target_name + "_" + training_model + ".pkl"

##################### End of Initialization #####################

########################### Main Program ###########################
#print ("test")

train_list = []
word2vec_input = []
pred = []

# train_data_file = "/media/amir/Amir/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized.csv"
# train_data_file = "sample_8000.csv"
# train_data_file = "sample_2000.csv"
#train_data_file = "sample100.csv"
#train_data_file = "/dev/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized_3000.csv"

#train_data_file = "/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized_3000.csv"
# train_data_file = "/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized_3.csv"


#baselines
# trained_41_exploratory.csv
# train_data_file =  "/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_17_300.csv"
train_data_file = "/dev/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_17_300.csv"
train_data_file = "/dev/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_17.csv"

#train_data = pd.read_csv(train_data_file, hsignedeader=0, delimiter="\t", encoding="utf-8", quoting=3)
train_data = pd.read_csv(train_data_file, delimiter="\t", encoding="utf-8", quoting=3, engine='python', error_bad_lines=False )
train_data=train_data.dropna()
#print train_data.head()


del train_data['Unnamed: 0']

# print train_data.shape
# print train_data['Gender'].describe()

# sys.exit()
#Comment for running the baselines
'''
del train_data['Unnamed..0.1']

#Unnamed..0.1
# add_numberical_features(train_data)
# sys.exit()
#print (pd.isnull(train_data))
#sys.exit()

#print (train_data.tail(2000))
#sys.exit()
#train_data.columns = ['id', 'sentiment', 'content','textBlob','pos_counts','neg_counts','negative_hashtag_count','positive_hashtag_count']
# train_data.columns = ["","screen_name","profile_background_color","profile_sidebar_border_color","profile_text_color","profile_sidebar_fill_color","followers_count","listed_count","description","statuses_count","friends_count","profile_link_color","favourites_count","lev_distance","description_sentiment_polarity","description_sentiment_subjectivity","content","combined_tweets_polarity","combined_tweets_subjectivity","pigeo_results_country","pigeo_results_city","pigeo_results_lon","pigeo_results_state","pigeo_results_lat","Annotation","avg_tweet_polarity","avg_tweet_subjectivity","avg_tweet_favorite_count","avg_tweet_retweet_count","profile_OCR_ocrText","profile_imageFeatures_averageRGB","profile_imageFeatures_hueVariance","profile_imageFeatures_nRedChannelMean","profile_imageFeatures_nGreenChannelMean","profile_imageFeatures_hueChannelMean","profile_imageFeatures_colorfulness","profile_imageFeatures_sharpness","profile_imageFeatures_brightness","profile_imageFeatures_saturationChannelVAR","profile_imageFeatures_ruleOfThirds","profile_imageFeatures_saturationChannelMean","profile_imageFeatures_naturalness","profile_imageFeatures_ocrText","profile_imageFeatures_hueChannelVAR","profile_imageFeatures_nBlueChannelMean","profile_imageFeatures_blur","profile_imageFeatures_meanHue","profile_imageFeatures_nGrayScaleMean","profile_imageFeatures_contrast","avg_tweet_imageFeatures_averageRGB","avg_tweet_imageFeatures_blur","avg_tweet_imageFeatures_brightness","avg_tweet_imageFeatures_colorfulness","avg_tweet_imageFeatures_contrast","avg_tweet_imageFeatures_hueChannelMean","avg_tweet_imageFeatures_hueChannelVAR","avg_tweet_imageFeatures_hueVariance","avg_tweet_imageFeatures_meanHue","avg_tweet_imageFeatures_nBlueChannelMean","avg_tweet_imageFeatures_nGrayScaleMean","avg_tweet_imageFeatures_nGreenChannelMean","avg_tweet_imageFeatures_nRedChannelMean","avg_tweet_imageFeatures_naturalness","avg_tweet_imageFeatures_ruleOfThirds","avg_tweet_imageFeatures_saturationChannelMean","avg_tweet_imageFeatures_saturationChannelVAR","avg_tweet_imageFeatures_sharpness","facepp_img_results_ES_faces_0_attributes_age_value","facepp_img_results_ES_faces_0_attributes_blur_blurness_threshold","facepp_img_results_ES_faces_0_attributes_blur_blurness_value","facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_threshold","facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_value","facepp_img_results_ES_faces_0_attributes_blur_motionblur_threshold","facepp_img_results_ES_faces_0_attributes_blur_motionblur_value","facepp_img_results_ES_faces_0_attributes_emotion_anger","facepp_img_results_ES_faces_0_attributes_emotion_disgust","facepp_img_results_ES_faces_0_attributes_emotion_fear","facepp_img_results_ES_faces_0_attributes_emotion_happiness","facepp_img_results_ES_faces_0_attributes_emotion_neutral","facepp_img_results_ES_faces_0_attributes_emotion_sadness","facepp_img_results_ES_faces_0_attributes_emotion_surprise","facepp_img_results_ES_faces_0_attributes_ethnicity_value","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_dark_glasses","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_no_glass_eye_close","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_no_glass_eye_open","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_normal_glass_eye_close","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_normal_glass_eye_open","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_occlusion","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_dark_glasses","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_no_glass_eye_close","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_no_glass_eye_open","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_normal_glass_eye_close","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_normal_glass_eye_open","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_occlusion","facepp_img_results_ES_faces_0_attributes_facequality_threshold","facepp_img_results_ES_faces_0_attributes_facequality_value","facepp_img_results_ES_faces_0_attributes_gender_value","facepp_img_results_ES_faces_0_attributes_glass_value","facepp_img_results_ES_faces_0_attributes_headpose_pitch_angle","facepp_img_results_ES_faces_0_attributes_headpose_roll_angle","facepp_img_results_ES_faces_0_attributes_headpose_yaw_angle","facepp_img_results_ES_faces_0_attributes_skinstatus_acne","facepp_img_results_ES_faces_0_attributes_skinstatus_dark_circle","facepp_img_results_ES_faces_0_attributes_skinstatus_health","facepp_img_results_ES_faces_0_attributes_skinstatus_stain","facepp_img_results_ES_faces_0_attributes_smile_threshold","facepp_img_results_ES_faces_0_attributes_smile_value","facepp_img_results_ES_faces_0_face_rectangle_height","facepp_img_results_ES_faces_0_face_rectangle_left","facepp_img_results_ES_faces_0_face_rectangle_top","facepp_img_results_ES_faces_0_face_rectangle_width","facepp_img_results_ES_faces_0_face_token","facepp_img_results_ES_request_id","facepp_img_results_ES_time_used","media_images_ocr","media_images_ocr_corrected","microsoft_img_results_ES_faceAttributes_age","microsoft_img_results_ES_faceAttributes_blur_blurLevel","microsoft_img_results_ES_faceAttributes_blur_value","microsoft_img_results_ES_faceAttributes_emotion_anger","microsoft_img_results_ES_faceAttributes_emotion_contempt","microsoft_img_results_ES_faceAttributes_emotion_disgust","microsoft_img_results_ES_faceAttributes_emotion_fear","microsoft_img_results_ES_faceAttributes_emotion_happiness","microsoft_img_results_ES_faceAttributes_emotion_neutral","microsoft_img_results_ES_faceAttributes_emotion_sadness","microsoft_img_results_ES_faceAttributes_emotion_surprise","microsoft_img_results_ES_faceAttributes_exposure_exposureLevel","microsoft_img_results_ES_faceAttributes_exposure_value","microsoft_img_results_ES_faceAttributes_facialHair_beard","microsoft_img_results_ES_faceAttributes_facialHair_moustache","microsoft_img_results_ES_faceAttributes_facialHair_sideburns","microsoft_img_results_ES_faceAttributes_gender","microsoft_img_results_ES_faceAttributes_glasses","microsoft_img_results_ES_faceAttributes_hair_bald","microsoft_img_results_ES_faceAttributes_hair_hairColor_0_color","microsoft_img_results_ES_faceAttributes_hair_hairColor_0_confidence","microsoft_img_results_ES_faceAttributes_hair_hairColor_1_color","microsoft_img_results_ES_faceAttributes_hair_hairColor_1_confidence","microsoft_img_results_ES_faceAttributes_hair_hairColor_2_color","microsoft_img_results_ES_faceAttributes_hair_hairColor_2_confidence","microsoft_img_results_ES_faceAttributes_hair_hairColor_3_color","microsoft_img_results_ES_faceAttributes_hair_hairColor_3_confidence","microsoft_img_results_ES_faceAttributes_hair_hairColor_4_color","microsoft_img_results_ES_faceAttributes_hair_hairColor_4_confidence","microsoft_img_results_ES_faceAttributes_hair_hairColor_5_color","microsoft_img_results_ES_faceAttributes_hair_hairColor_5_confidence","microsoft_img_results_ES_faceAttributes_hair_invisible","microsoft_img_results_ES_faceAttributes_headPose_pitch","microsoft_img_results_ES_faceAttributes_headPose_roll","microsoft_img_results_ES_faceAttributes_headPose_yaw","microsoft_img_results_ES_faceAttributes_makeup_eyeMakeup","microsoft_img_results_ES_faceAttributes_makeup_lipMakeup","microsoft_img_results_ES_faceAttributes_noise_noiseLevel","microsoft_img_results_ES_faceAttributes_noise_value","microsoft_img_results_ES_faceAttributes_occlusion_eyeOccluded","microsoft_img_results_ES_faceAttributes_occlusion_foreheadOccluded","microsoft_img_results_ES_faceAttributes_occlusion_mouthOccluded","microsoft_img_results_ES_faceAttributes_smile","microsoft_img_results_ES_faceId","microsoft_img_results_ES_faceRectangle_height","microsoft_img_results_ES_faceRectangle_left","microsoft_img_results_ES_faceRectangle_top","microsoft_img_results_ES_faceRectangle_width"]
# train_data.content.to_csv("unlabeledTrainData.csv", sep='\t')

#mergin Twitter desciption and content together:
train_data['merged_tweets_description'] = train_data['combined_tweets'].astype(str) + train_data['description'].astype(str)
#names = list(train_data.columns.values)
#print names
train_data.drop(columns= ['Unnamed: 0', u'Gender', u'profile_imageFeatures.BlueChannelMean', u'profile_imageFeatures.GrayScaleMean', u'profile_imageFeatures.GreenChannelMean', u'profile_imageFeatures.colorfulness', u'profile_imageFeatures.contrast', u'profile_imageFeatures.hueChannelMean', u'profile_imageFeatures.hueChannelVAR', u'profile_imageFeatures.naturalness', u'profile_imageFeatures.saturationChannelMean', u'profile_imageFeatures.saturationChannelVAR', u'profile_imageFeatures.sharpness', u'tweet_imageFeatures.BlueChannelMean', u'tweet_imageFeatures.GrayScaleMean', u'tweet_imageFeatures.GreenChannelMean', u'tweet_imageFeatures.RedChannelMean', u'tweet_imageFeatures.averageRGB', u'tweet_imageFeatures.brightness', u'tweet_imageFeatures.colorfulness', u'tweet_imageFeatures.contrast', u'tweet_imageFeatures.hueChannelMean', u'tweet_imageFeatures.hueChannelVAR', u'tweet_imageFeatures.naturalness', u'tweet_imageFeatures.saturationChannelMean', u'tweet_imageFeatures.saturationChannelVAR', u'tweet_imageFeatures.sharpness', u'Analytic', u'Authentic', u'Clout', u'Dic', u'Sixltr', u'Tone', u'WC', u'achieve', u'adj', u'adverb', u'affect', u'affiliation', u'anger', u'anx', u'article', u'assent', u'auxverb', u'bio', u'body', u'cause', u'certain', u'cogproc', u'compare', u'conj', u'death', u'differ', u'discrep', u'drives', u'family', u'feel', u'female', u'focusfuture', u'focuspast', u'focuspresent', u'friend', u'function.', u'health', u'hear', u'home', u'i', u'informal', u'ingest', u'insight', u'interrog', u'ipron', u'leisure', u'male', u'money', u'motion', u'negate', u'negemo', u'netspeak', u'nonflu', u'number', u'percept', u'posemo', u'power', u'ppron', u'prep', u'pronoun', u'quant', u'relativ', u'relig', u'reward', u'risk', u'sad', u'see', u'sexual', u'shehe', u'social', u'space', u'swear', u'tentat', u'they', u'time', u'verb', u'we', u'work', u'you', u'profile_background_color', u'profile_sidebar_border_color', u'profile_text_color', u'profile_sidebar_fill_color', u'followers_count', u'description', u'statuses_count', u'friends_count', u'profile_link_color', u'favourites_count', u'lev_distance', u'description_sentiment_polarity', u'description_sentiment_subjectivity', u'combined_tweets', u'combined_tweets_polarity', u'combined_tweets_subjectivity', u'pigeo_results_lat', u'avg_tweet_polarity', u'avg_tweet_subjectivity', u'avg_tweet_favorite_count', u'avg_tweet_retweet_count', u'face_pp_type', u'facepp_img_results_ES_faces_0_attributes_blur_blurness_value', u'facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_value', u'facepp_img_results_ES_faces_0_attributes_blur_motionblur_value', u'facepp_img_results_ES_faces_0_attributes_emotion_anger', u'facepp_img_results_ES_faces_0_attributes_emotion_disgust', u'facepp_img_results_ES_faces_0_attributes_emotion_fear', u'facepp_img_results_ES_faces_0_attributes_emotion_happiness', u'facepp_img_results_ES_faces_0_attributes_emotion_neutral', u'facepp_img_results_ES_faces_0_attributes_emotion_sadness', u'facepp_img_results_ES_faces_0_attributes_emotion_surprise', u'facepp_img_results_ES_faces_0_attributes_ethnicity_value', u'facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_no_glass_eye_open', u'facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_no_glass_eye_open', u'facepp_img_results_ES_faces_0_attributes_facequality_value', u'facepp_img_results_ES_faces_0_attributes_gender_value', u'facepp_img_results_ES_faces_0_attributes_headpose_pitch_angle', u'facepp_img_results_ES_faces_0_attributes_headpose_roll_angle', u'facepp_img_results_ES_faces_0_attributes_headpose_yaw_angle', u'facepp_img_results_ES_faces_0_attributes_skinstatus_dark_circle', u'facepp_img_results_ES_faces_0_attributes_skinstatus_health', u'facepp_img_results_ES_faces_0_attributes_skinstatus_stain', u'facepp_img_results_ES_faces_0_attributes_smile_value', u'facepp_img_results_ES_faces_0_face_rectangle_height', u'facepp_img_results_ES_faces_0_face_rectangle_left', u'facepp_img_results_ES_faces_0_face_rectangle_top', u'facepp_img_results_ES_faces_0_face_rectangle_width', u'Human.Judge.for.Gender', u'Age_text_cat', u'Human.judgement.for.age_cat', u'facepp_img_results_ES_faces_0_attributes_age_value_cat'])
# sys.exit()
'''

print vector_type
# sys.exit()

if vector_type == "Word2vec":
    #unlab_train_data = pd.read_csv("trained_data_for_model_median_impu_normalized.csv", header=0, quoting=3)
    unlab_train_data = train_data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


# Extract words from documents
# range is faster when iterating
if vector_type == "Word2vec" or vector_type == "Word2vec_pretrained":

    for i in range(0, len(train_data.combined_tweets)):
        if vector_type == "Word2vec":
            # Decode utf-8 coding first
            word2vec_input.extend(document_to_doublelist(train_data['combined_tweets'].iloc[i].decode("utf-8"), tokenizer))

        train_list.append(clean_document(train_data['combined_tweets'].iloc[i], output_format="list"))
        if i%1000 == 0:
            print ("Cleaning training content", i)

    if vector_type == "Word2vec":
        for i in range(0, len(unlab_train_data['combined_tweets'])):
            word2vec_input.extend(document_to_doublelist(unlab_train_data['combined_tweets'].iloc[i].decode("utf-8"), tokenizer))
            if i%1000 == 0:
                print ("Cleaning unlabeled training content", i)


elif vector_type != "no":

    for i in range(0, len(train_data.merged_tweets_description)):
        # Append raw texts rather than lists as Count/TFIDF vectorizers take raw texts as inputs

        try:

            train_list.append(clean_document(train_data['merged_tweets_description'].iloc[i]))
            #print ("trainlist--->",train_list[i])

        except Exception as e:
            raise
    #print "train_list[0]",train_list [0]



# Generate vectors from words
if vector_type == "Word2vec_pretrained" or vector_type == "Word2vec":

    if vector_type == "Word2vec_pretrained":
        print ("Loading the pre-trained model")
        if model_type == "bin":
            model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True,unicode_errors='ignore')

        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(model_name,unicode_errors='ignore')

    if vector_type == "Word2vec":
        print ("Training word2vec word vectors")
        model = gensim.models.Word2Vec(word2vec_input, workers=num_workers,size=num_features, min_count = min_word_count,window = context, sample = downsampling)
        # print dir(model)
        # sys.exit()
        # If no further training and only query is needed, this trims unnecessary memory
        model.init_sims(replace=True)

        # Save the model for later use
        model.save(model_name)

    print ("Vectorizing training content via embeddings")
    print "train_list", len (train_list)
    train_vec = gen_document_vecs(train_list, model, num_features)

elif sent2vec_flag:
    #print "heressssssssssssssssssss", str(vector_type)

    if str(vector_type) == "word2vec2" and train_embedding_flag == False:
        embeddings_index = load_word2vec_dic(model_name)

        train_vec = [sent2vec(x,embeddings_index) for x in tqdm(train_list)]
        train_vec = np.array(train_vec)
    if str(vector_type) == "word2vec2" and train_embedding_flag == True:
        #unlabaled data
        input_file = "/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/unlabeledTrainData.csv"
        documents = read_input(input_file)
        sentences = SentencesIterator(documents)
        #print len(documents)
        #sys.exit()
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            sentences,
            size=300,
            window=10,
            min_count=2,
            workers=10)
        model_trained_on_our_data = model.train(documents, total_examples=len(documents), epochs=10)
        if embedding_model_checking:
            w1 = "dirty"
            print model.wv.most_similar(positive = w1)

elif topic_modeling_based_reduction:
    print "in Topic modeling based reduction"
    #if topic_base_addition:
    print "Vectorizing..."
    count_vec = CountVectorizer(analyzer="word",min_df = 5, max_df=0.8,binary = (vector_type == "Binary"),ngram_range=(1,3), lowercase= True)
    train_vec = count_vec.fit_transform(train_list)
    print "train_vec shape...",train_vec.shape

    #vector_type_old = vector_type
    #vector_type  = "LDA" # changing vector type to add
    #print "current vector_type: ", vector_type
    if vector_type == "LDA" :
        print "LDA will start"
        # train a LDA Model
        lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, learning_method='online', max_iter=50)
        X_topics = lda_model.fit_transform(train_vec)
        print X_topics.shape
        #show the first element
        print X_topics[0]

        print("LDA Model:")
        print_topics(lda_model, count_vec)
        print("=" * 20)

        #for col_name in X_topics:
         #   col_name_values = train_data[[str(col_name)]].values
          #  train_vec = sparse.hstack((train_vec, col_name_values.astype(float))).tocsr()

        #train_vec = sparse.hstack((train_vec, X_topics.astype(float))).tocsr()
        #print("train_vec.shape after topics: ", train_vec.shape)

        #train_vec = sparse.hstack((train_vec, col_name_values.astype(float))).tocsr()

        #print("train_vec.shape after adding numerical values: ", train_vec.shape)

        train_vec = X_topics

    elif vector_type == "NMF":
        # Build a Non-Negative Matrix Factorization Model
        nmf_model = NMF(n_components=NUM_TOPICS)
        nmf_Z = nmf_model.fit_transform(train_vec)
        print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

        print("NMF Model:")
        print_topics(nmf_model, count_vec)
        print("=" * 20)
        train_vec = nmf_Z
    elif vector_type == "LSI":
        # Build a Latent Semantic Indexing Model
        lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
        lsi_Z = lsi_model.fit_transform(train_vec)
        print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
        print("LSI Model:")
        print_topics(lsi_model, count_vec)
        print("=" * 20)
        train_vec = lsi_Z
    #vector_type = vector_type_old

# elif vector_type != "no" and vector_type  == "TFIDF" or  vector_type  == "Binary" or vector_type  == "Int" :
if vector_type != "no" :
    print "in Tf_idf"
    if vector_type == "TFIDF":
        # Unit of gshow()ram is "word", only top 5000/10000 words are extracted
        count_vec = TfidfVectorizer(analyzer="word", max_features=100, ngram_range=(1,3), sublinear_tf=True)
        #count_vec._validate_vocabulary()
        #TF_IDF_feature_names = count_vec.get_feature_names()


    elif vector_type == "Binary" or vector_type == "Int":
        count_vec = CountVectorizer(analyzer="word", max_features=100,binary = (vector_type == "Binary"),ngram_range=(1,3))
        #count_vec._validate_vocabulary()
        #binary_feature_names =  count_vec.get_feature_names()

        #print('count_vec.get_feature_names(): {0}'.format(count_vec.get_feature_names()))
        #sys.exit()
    # Return a scipy sparse term-document matrix
    print ("Vectorizing input texts")

    train_vec = count_vec.fit_transform(train_list)
    feature_names = count_vec.get_feature_names()
    # print feature_names
    # sys.exit()
    #train_vec._validate_vocabulary()
    #binary_feature_names =  train_vec.get_feature_names()


    # print binary_int_feature_names
    # sys.exit()
    # new_text_blob = train_data[['textBlob']].values
    # train_vec = sparse.hstack((train_vec, new_text_blob)).tocsr()
    #
    #################################   ###############
    # pos_counts = train_data[['pos_counts']].values
    # train_vec = sparse.hstack((train_vec, pos_counts)).tocsr()
    #
    # neg_counts = train_data[['neg_counts']].values
    # train_vec = sparse.hstack((train_vec, neg_counts)).tocsr()
    # ######################################################


    # pos_counts_hashtags = train_data[['positive_hashtag_count']].values
    # train_vec = sparse.hstack((train_vec, pos_counts_hashtags)).tocsr()
    #
    # neg_counts_hashtags = train_data[['negative_hashtag_count']].values
    # train_vec = sparse.hstack((train_vec, neg_counts_hashtags)).tocsr()

    #"profile_background_color","profile_sidebar_border_color","profile_text_color"


    #adding image features
    train_vec, numerical_col_names = add_numberical_features(train_data,train_vec)
    feature_names = feature_names + numerical_col_names
    feature_names = feature_names
    print ("train_vec.shape",train_vec.shape)


    if topic_base_addition:
        train_vec = add_topical_feature(train_list,train_vec,topical_type)
        print "train_vec shape after adding topics as feature",train_vec.shape
    #add natural_langauge features
    if add_nlp_feature_flag:

        train_data = add_nlp_feature (train_data)
        print ("train_vec.shape before adding nlp fetures:",train_vec.shape)
        train_vec, numerical_col_names = add_numberical_features(train_data,train_vec)
        feature_names = feature_names + numerical_col_names
        print ("train_vec.shape after adding nlp fetures:",train_vec.shape)
        # train_vec = train_data # for only keeping NLP features

if scaling != "no":
    #max_abs_scaler = preprocessing.MaxAbsScaler(max_abs_scaler = preprocessing.MaxAbsScaler())
    if scaling == "standard":
        scaler = preprocessing.StandardScaler(with_mean=False)
    else:
        if scaling == "unsigned":
            #scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
            scaler = preprocessing.MaxAbsScaler()

        elif scaling == "signed":
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    print ("Scaling vectors")
    train_vec = scaler.fit_transform(train_vec)
    print (train_vec)

    #sys.exit()

# Feature Scaling
# Dimemsion Reduction
if dim_reduce == "SVD":
    print ("Performing dimension reduction")
    svd = TruncatedSVD(n_components = num_dim)
    train_vec = svd.fit_transform(train_vec)
    print ("Explained variance ratio =", svd.explained_variance_ratio_.sum())

elif dim_reduce == "chi2":
    print ("Performing feature selection based on chi2 independence test")
    chi2score = chi2(train_vec,train_data.Annotation)[0]
    ##################
    fselect = SelectKBest(chi2, k = num_dim)
    train_vec = fselect.fit_transform(train_vec, train_data.Annotation)
    #print (train_vec.get_feature_names())

    '''
    figure(figsize=(6,6))
    wscores = zip(count_vec.get_feature_names(),chi2score)
    wchi2 = sorted(wscores,key=lambda x:x[1])
    topchi2 = list(zip(*wchi2[-100:]))
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    barh(x,topchi2[1],align='center',alpha=.2,color='g')
    plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
    yticks(x,labels)
    xlabel('$\chi^2$')
    #show()
    '''
    #train_vec = add_numberical_features(train_data,train_vec)
    #
    #description_sentiment_polarity = train_data[['description_sentiment_polarity']].values
    #train_vec = sparse.hstack((train_vec, description_sentiment_polarity.astype(float))).tocsr()
    #

    print ("train_vec.shape",train_vec.shape)

# Transform into numpy arrays
if "numpy.ndarray" not in str(type(train_vec)):
    train_vec = train_vec.toarray()

# Model training
if training_model == "RF" or training_model == "BT":
    print("train_vec.shape before fit the models", train_vec.shape)
    # Initialize the Random Forest or bagged tree based the model chosen
    #max_features: These are the maximum number of features Random Forest is allowed to try in individual tree , Increasing max_features generally improves the performance of the model as at each node now we have a higher number of options to be considered, for sure, you decrease the speed of algorithm by increasing the max_features. max_features = [sqrt,0.2 ]
    #n_estimator: This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower.
    #oob_score: tags every observation used in different tress. And then it finds out a maximum vote score for every observation based on only trees which did not use this particular observation to train itself.
    #min_sample_leaf : . A smaller leaf makes the model more prone to capturing noise in train data. Generally I prefer a minimum leaf size of more than 50,foe istance: [ 50,100,200,500]
    rfc = RFC(n_estimators = 100, oob_score = True,max_features = (None if training_model=="BT" else "auto"), min_samples_leaf = 100)
    cv_accuracy = cross_val_score(rfc, train_vec, train_data.Annotation, scoring="accuracy", cv=10)
    cv_prec = cross_val_score(rfc, train_vec, train_data.Annotation, scoring="precision_macro",  cv=10)
    cv_rec = cross_val_score(rfc, train_vec, train_data.Annotation, scoring="recall_macro", cv=10)
    cv_f1 = cross_val_score(rfc, train_vec, train_data.Annotation, scoring="f1_macro", cv=10)
    print ("Training %s" % ("Random Forest" if training_model=="RF" else "bagged tree"))
    print ("CV Accuracy = %.4f" % cv_accuracy.mean())
    print ("CV Precision = %.4f" % cv_prec.mean())
    print ("CV Recall = %.4f" % cv_rec.mean())
    print ("CV F1 Score = %.4f" % cv_f1.mean())
    rfc = rfc.fit(train_vec, train_data.Annotation)
    print ("OOB Score =", rfc.oob_score_)
    if save_model:
        f = open(model_file, "wb")
        pickle.dump(rfc, f)

    if learning_curve_flag:
        fitted_model = rfc
        train_data_Annotation = train_data.Annotation
        draw_learning_cuve(fitted_model,train_vec,train_data_Annotation)

    if show_feature:
        print "Feature Visualization!!!"
        show_feature_function(rfc,train_vec )

        #feature_importance_with_forrest(rfc,train_vec )

    print "Feature importance base on Ensemble trees: "
    print (rfc.feature_importances_)
elif training_model == "NB":
    print("train_vec.shape before fit the models", train_vec.shape)
    print("Type Train_vec---->", type(train_vec))
    nb = naive_bayes.MultinomialNB()
    cv_accuracy = cross_val_score(nb, train_vec, train_data.Annotation, scoring="accuracy", cv=10)
    cv_prec = cross_val_score(nb, train_vec, train_data.Annotation, scoring="precision_macro",  cv=10)
    cv_rec = cross_val_score(nb, train_vec, train_data.Annotation, scoring="recall_macro", cv=10)
    cv_f1 = cross_val_score(nb, train_vec, train_data.Annotation, scoring="f1_macro", cv=10)
    print ("Training Naive Bayes")
    print ("CV Accuracy = %.4f" % cv_accuracy.mean())
    print ("CV Precision = %.4f" % cv_prec.mean())
    print ("CV Recall = %.4f" % cv_rec.mean())
    print ("CV F1 Score = %.4f" % cv_f1.mean())
    nb = nb.fit(train_vec, train_data.Annotation)
#    f = open("../models/edrugtrend_nb_model_sentiment.pkl","wb")
    if save_model:
        f = open(model_file, "wb")
        pickle.dump(nb, f)

elif training_model == "LR":
    print("train_vec.shape before fit the models", train_vec.shape)
    lr = linear_model.LogisticRegression(dual=False,class_weight="balanced", solver='lbfgs', multi_class="multinomial")
    cv_accuracy = cross_val_score(lr, train_vec, train_data.Annotation, scoring="accuracy", cv=10)
    cv_prec = cross_val_score(lr, train_vec, train_data.Annotation, scoring="precision_macro",  cv=10)
    cv_rec = cross_val_score(lr, train_vec, train_data.Annotation, scoring="recall_macro", cv=10)
    cv_f1 = cross_val_score(lr, train_vec, train_data.Annotation, scoring="f1_macro", cv=10)
    print ("Training Logistic Regression")
    print ("CV Accuracy = %.4f" % cv_accuracy.mean())
    print ("CV Precision = %.4f" % cv_prec.mean())
    print ("CV Recall = %.4f" % cv_rec.mean())
    print ("CV F1 Score = %.4f" % cv_f1.mean())
    lr = lr.fit(train_vec, train_data.Annotation)
    if save_model:
        f = open(model_file, "wb")
        pickle.dump(lr, f)

    if learning_curve_flag:
        fitted_model = lr
        print "Drawing Learning Curve"
        train_data_Annotation = train_data.Annotation
        draw_learning_cuve(fitted_model,train_vec,train_data_Annotation)

    if show_feature:
        #show_feature_function(svc,train_vec )
        #feature_importance_with_forrest()
        print "Feature Visualization!!!"
        feature_visualization(lr,feature_names)
elif training_model == "SVM":
    print("train_vec.shape before fit the models", train_vec.shape)
    svc = svm.LinearSVC(C=1.0)
#    svc = svm.SVC(C=1.0)
#    param = {'C': [1e15,1e13,1e11,1e9,1e7,1e5,1e3,1e1,1e-1,1e-3,1e-5]}
    print ("Training SVM")
#    f1_macro = metrics.make_scorer(metrics.f1_score, pos_label=None, average='macro')
#    svc = GridSearchCV(svc, param, score_func = metrics.accuracy_score, cv=5)
    cv_accuracy = cross_val_score(svc, train_vec, train_data.Annotation, scoring="accuracy", cv=10)
    cv_prec = cross_val_score(svc, train_vec, train_data.Annotation, scoring="precision_macro",  cv=10)
    cv_rec = cross_val_score(svc, train_vec, train_data.Annotation, scoring="recall_macro", cv=10)
    cv_f1 = cross_val_score(svc, train_vec, train_data.Annotation, scoring="f1_macro", cv=10)
    print ("Training Support Vector Machine")
    print ("CV Accuracy = %.4f" % cv_accuracy.mean())
    print ("CV Precision = %.4f" % cv_prec.mean())
    print ("CV Recall = %.4f" % cv_rec.mean())
    print ("CV F1 Score = %.4f" % cv_f1.mean())
    show()
    svc = svc.fit(train_vec, train_data.Annotation)
#    print ("Optimized parameters:", svc.best_estimator_)
#    print ("Best CV score:", svc.best_score_)
    if save_model:
        f = open(model_file, "wb")
        pickle.dump(svc, f)


    if learning_curve_flag:
        fitted_model = svc
        print "Drawing Learning Curve"
        train_data_Annotation = train_data.Annotation
        draw_learning_cuve(fitted_model,train_vec,train_data_Annotation)

    if show_feature:
        #show_feature_function(svc,train_vec )
        #feature_importance_with_forrest()
        print "Feature Visualization!!!"
        feature_visualization(svc,feature_names)
