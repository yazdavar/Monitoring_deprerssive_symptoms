
# coding: utf-8

# In[1]:


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

############################################################-----------------------------------
#import os
import re
import logging
import pandas as pd
import numpy as np
import nltk.data
import sys
# import sklearn
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.preprocessing import Imputer
from sklearn import linear_model, naive_bayes, svm, preprocessing
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
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
import emoji
import seaborn as sns
#import requests
from sklearn.feature_selection import SelectKBest, f_classif


from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, TransformerMixin





from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#import xgboost as xgb

##################### Function Definition #####################


def clean_document(document, remove_stopwords = False, output_format = "string"):
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
    text = BeautifulSoup(document)

    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())

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
    index2word_set = set(model.index2word)

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
    #print numerical_col_names
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


def feature_visualization(classifier, feature_names, top_features=20):
    '''
    This is the function which visulize both numerical and textual coef
    for each class of positive and negative
    '''
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    fig =plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    # print top_coefficients
    # print feature_names
    # sys.exit()
    print (feature_names[1:50])
    print len(feature_names)
    print top_coefficients
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
    #plt.show()
    fig.savefig('feature_Coef.png')



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
    #plt.bar(range (20), importances[indices], color="r", yerr=std[selected_indices], align="center")
    plt.bar(range (20), selected, color="r", align="center")
    # plt.bar(range (20), selected, color="r", align="center")
    #plt.xticks(range(train_vec.shape[1]), indices)
    plt.xticks(range(20), selected_indices)
    # plt.xlim([-1, train_vec.shape[1]])
    plt.xlim([-1, 20])
    #plt.show()
    fig.savefig('feature_Random_forrest.png')


def explore_target(train_data):
    '''
    function  for visulizaing the class attibute
    '''

    sns.factorplot(x="Annotation", data=train_data, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")
    fig = plt.figure()
    plt.show()
    fig.savefig('class_distribution.png')


class ExploreText(BaseEstimator, TransformerMixin):

    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))

    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self

    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x))
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))

        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital_words': count_capital_words
                           , 'count_excl_quest_marks': count_excl_quest_marks
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })

        return df



def show_dist(df, col):
    print('Descriptive stats for {}'.format(col))
    print('-'*(len(col)+22))
    print(df.groupby('Annotation')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='Annotation', size=5, hue='Annotation', palette="PuBuGn_d")
    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)
    fig = plt.figure()
    plt.show()
    fig.savefig('dist.png')


class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)

    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')

    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)

    def to_lower(self, input_text):
        return input_text.lower()

    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X
##################### End of Function Definition #####################

##################### Initialization #####################

# term_vector_type = {"TFIDF", "Binary", "Int", "Word2vec", "Word2vec_pretrained"}
# {"TFIDF", "Int", "Binary"}: Bag-of-words model with {tf-idf, word counts, presence/absence} representation
# {"Word2vec", "Word2vec_pretrained"}: Google word2vec representation {without, with} pre-trained models
# Specify model_name if there's a pre-trained model to be loaded
vector_type = "Word2vec"
model_name = "GoogleNews-vectors-negative300.bin"


learning_curve_flag = True

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
training_model = "SVM"

#show feature Importance
show_feature = True

# feature scaling = {"standard", "signed", "unsigned", "no"}
# Note: Scaling is needed for SVM
scaling = "standard"

# dimension reduction = {"SVD", "chi2", "no"}
# Note: For NB models, we cannot perform truncated SVD as it will make input negative
# chi2 is the feature selectioin based on chi2 independence test
dim_reduce = "no"
num_dim = 500

# campaign, and target entity
campaign_name = "election2016"
target_name = "hillary_clinton_sentiment_counts_pos_negs_hashtags_pos_neg_new"
# train data file
train_data_file = "../data/" + campaign_name + "/" + target_name + ".csv"
# model file
save_model = False
model_file = "../models/" + campaign_name + "/" + target_name + "_" + training_model + ".pkl"

##################### End of Initialization #####################


# In[ ]:


########################### Main Program ###########################
#print ("test")

train_list = []
word2vec_input = []
pred = []
train_data_file = "/media/amir/Amir/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized.csv"
train_data_file = "/media/amir/Amir/Image_analysis_depressin_datasets/test/sample100.csv"
# train_data_file = "/media/amir/Amir/Image_analysis_depressin_datasets/merged_tweet_imageFeatures_nan_handled.csv"
#train_data = pd.read_csv(train_data_file, hsignedeader=0, delimiter="\t", encoding="utf-8", quoting=3)
train_data = pd.read_csv(train_data_file, delimiter="\t", encoding="utf-8", quoting=3, engine='python', error_bad_lines=False )
train_data=train_data.dropna()
print train_data.head()

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


#--------------------Exploring emoji and hashtags, urls,  captial words provide stat and visulization----------------------------
#explore_target(train_data)
#et = ExploreText()
#df_eda = et.fit_transform(train_data.merged_tweets_description)


# Add annotaion to df_eda
#df_eda['Annotation'] = train_data.Annotation

# show_dist(df_eda, 'count_words')
# show_dist(df_eda, 'count_hashtags')
# show_dist(df_eda, 'count_emojis')
# show_dist(df_eda, 'count_urls')
# show_dist(df_eda, 'count_excl_quest_marks')
# show_dist(df_eda, 'count_capital_words')
#show_dist(df_eda, 'count_mentions')


#--------------------- harsh cleaing on the data---------------------------
# ct = CleanText()
# sr_clean = ct.fit_transform(train_data.merged_tweets_description)
# print sr_clean.sample(5)

# sys.exit()

if vector_type == "Word2vec":
    data_for_word2vec = "/media/amir/Amir/Image_analysis_depressin_datasets/unlabeledTrainData.csv"
    train_data = pd.read_csv(data_for_word2vec, header=0, delimiter="\t", quoting=3)
    train_data.columns = ["", "merged_tweets_description"]
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


# Extract words from documents
# range is faster when iterating
if vector_type == "Word2vec" or vector_type == "Word2vec_pretrained":

    for i in range(0, len(train_data.merged_tweets_description)):
        if vector_type == "Word2vec":
            # Decode utf-8 coding first
            word2vec_input.extend(document_to_doublelist(train_data.merged_tweets_description[i].decode("utf-8"), tokenizer))

        train_list.append(clean_document(train_data.merged_tweets_description[i], output_format="list"))
        if i%1000 == 0:
            print ("Cleaning training content", i)

    if vector_type == "Word2vec":
        for i in range(0, len(unlab_train_data.merged_tweets_description)):
            word2vec_input.extend(document_to_doublelist(unlab_train_data.merged_tweets_description[i].decode("utf-8"), tokenizer))
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


# Generate vectors from words
if vector_type == "Word2vec_pretrained" or vector_type == "Word2vec":

    if vector_type == "Word2vec_pretrained":
        print ("Loading the pre-trained model")
        if model_type == "bin":
            model = word2vec.Word2Vec.load_word2vec_format(model_name, binary=True)
        else:
            model = word2vec.Word2Vec.load(model_name)

    if vector_type == "Word2vec":
        print ("Training word2vec word vectors")
        model = word2vec.Word2Vec(word2vec_input, workers=num_workers,size=num_features, min_count = min_word_count,window = context, sample = downsampling)

        # If no further training and only query is needed, this trims unnecessary memory
        model.init_sims(replace=True)

        # Save the model for later use
        model.save(model_name)

    print ("Vectorizing training contennp.asarray(vectorizer.get_feature_names())[ch2.get_support()]t")
    train_vec = gen_document_vecs(train_list, model, num_features)



elif vector_type != "no":
    if vector_type == "TFIDF":
        # Unit of gshow()ram is "word", only top 5000/10000 words are extracted
        count_vec = TfidfVectorizer(analyzer="word", max_features=5000, ngram_range=(1,2), sublinear_tf=True)
        #count_vec._validate_vocabulary()
        #TF_IDF_feature_names = count_vec.get_feature_names()


    elif vector_type == "Binary" or vector_type == "Int":
        count_vec = CountVectorizer(analyzer="word", max_features=5000,binary = (vector_type == "Binary"),ngram_range=(1,2))
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
    ################################################
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


    print type (train_vec)
    sys.exit()
    train_vec, numerical_col_names = add_numberical_features(train_data,train_vec)
    feature_names = feature_names + numerical_col_names
    print ("train_vec.shape",train_vec.shape)


print type (train_vec)
# sys.exit()
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


    figure(figsize=(6,6))
    wscores = zip(count_vec.get_feature_names(),chi2score)
    wchi2 = sorted(wscores,key=lambda x:x[1])
    topchi2 = list(zip(*wchi2[-100:]))
    x = range(len(topchi2[1]))
    #print(len(topchi2[1]))
    #sys.exit()
    labels = topchi2[0]
    barh(x,topchi2[1],align='center',alpha=.2,color='g')
    plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
    yticks(x,labels)
    xlabel('$\chi^2$')
    #show()



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
    # Initialize the Random Forest or bagged tree based the model chosen
    rfc = RFC(n_estimators = 100, oob_score = True,max_features = (None if training_model=="BT" else "auto"))
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

    if show_feature:
        #show_feature_function(svc,train_vec )
        feature_importance_with_forrest(rfc,train_vec )

    if learning_curve_flag:
        fitted_model = rfc
        train_data_Annotation = train_data.Annotation
        draw_learning_cuve(fitted_model,train_vec,train_data_Annotation)


    print "Feature importance base on Ensemble trees: "
    print (rfc.feature_importances_)
elif training_model == "NB":
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
    lr = linear_model.LogisticRegression(dual=False,                 class_weight="balanced", solver='lbfgs', multi_class="multinomial")
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

elif training_model == "SVM":
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

    #print binary_int_feature_names

    if show_feature:
        #show_feature_function(svc,train_vec )
        #feature_importance_with_forrest()

        feature_visualization(svc,feature_names)

    if learning_curve_flag:
        fitted_model = svc
        print "here"
        train_data_Annotation = train_data.Annotation
        draw_learning_cuve(fitted_model,train_vec,train_data_Annotation)
