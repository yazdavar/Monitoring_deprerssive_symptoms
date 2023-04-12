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



train_data_file = "/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized_3000.csv"
#train_data_file = "/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_data_for_model_median_impu_normalized_10.csv"


train_data = pd.read_csv(train_data_file, delimiter="\t", encoding="utf-8", quoting=3, engine='python', error_bad_lines=False )
train_data=train_data.dropna()
#print train_data.head()
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
#train_data.drop(columns= ['Unnamed: 0', u'Gender', u'profile_imageFeatures.BlueChannelMean', u'profile_imageFeatures.GrayScaleMean', u'profile_imageFeatures.GreenChannelMean', u'profile_imageFeatures.colorfulness', u'profile_imageFeatures.contrast', u'profile_imageFeatures.hueChannelMean', u'profile_imageFeatures.hueChannelVAR', u'profile_imageFeatures.naturalness', u'profile_imageFeatures.saturationChannelMean', u'profile_imageFeatures.saturationChannelVAR', u'profile_imageFeatures.sharpness', u'tweet_imageFeatures.BlueChannelMean', u'tweet_imageFeatures.GrayScaleMean', u'tweet_imageFeatures.GreenChannelMean', u'tweet_imageFeatures.RedChannelMean', u'tweet_imageFeatures.averageRGB', u'tweet_imageFeatures.brightness', u'tweet_imageFeatures.colorfulness', u'tweet_imageFeatures.contrast', u'tweet_imageFeatures.hueChannelMean', u'tweet_imageFeatures.hueChannelVAR', u'tweet_imageFeatures.naturalness', u'tweet_imageFeatures.saturationChannelMean', u'tweet_imageFeatures.saturationChannelVAR', u'tweet_imageFeatures.sharpness', u'Analytic', u'Authentic', u'Clout', u'Dic', u'Sixltr', u'Tone', u'WC', u'achieve', u'adj', u'adverb', u'affect', u'affiliation', u'anger', u'anx', u'article', u'assent', u'auxverb', u'bio', u'body', u'cause', u'certain', u'cogproc', u'compare', u'conj', u'death', u'differ', u'discrep', u'drives', u'family', u'feel', u'female', u'focusfuture', u'focuspast', u'focuspresent', u'friend', u'function.', u'health', u'hear', u'home', u'i', u'informal', u'ingest', u'insight', u'interrog', u'ipron', u'leisure', u'male', u'money', u'motion', u'negate', u'negemo', u'netspeak', u'nonflu', u'number', u'percept', u'posemo', u'power', u'ppron', u'prep', u'pronoun', u'quant', u'relativ', u'relig', u'reward', u'risk', u'sad', u'see', u'sexual', u'shehe', u'social', u'space', u'swear', u'tentat', u'they', u'time', u'verb', u'we', u'work', u'you', u'profile_background_color', u'profile_sidebar_border_color', u'profile_text_color', u'profile_sidebar_fill_color', u'followers_count', u'description', u'statuses_count', u'friends_count', u'profile_link_color', u'favourites_count', u'lev_distance', u'description_sentiment_polarity', u'description_sentiment_subjectivity', u'combined_tweets', u'combined_tweets_polarity', u'combined_tweets_subjectivity', u'pigeo_results_lat', u'avg_tweet_polarity', u'avg_tweet_subjectivity', u'avg_tweet_favorite_count', u'avg_tweet_retweet_count', u'face_pp_type', u'facepp_img_results_ES_faces_0_attributes_blur_blurness_value', u'facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_value', u'facepp_img_results_ES_faces_0_attributes_blur_motionblur_value', u'facepp_img_results_ES_faces_0_attributes_emotion_anger', u'facepp_img_results_ES_faces_0_attributes_emotion_disgust', u'facepp_img_results_ES_faces_0_attributes_emotion_fear', u'facepp_img_results_ES_faces_0_attributes_emotion_happiness', u'facepp_img_results_ES_faces_0_attributes_emotion_neutral', u'facepp_img_results_ES_faces_0_attributes_emotion_sadness', u'facepp_img_results_ES_faces_0_attributes_emotion_surprise', u'facepp_img_results_ES_faces_0_attributes_ethnicity_value', u'facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_no_glass_eye_open', u'facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_no_glass_eye_open', u'facepp_img_results_ES_faces_0_attributes_facequality_value', u'facepp_img_results_ES_faces_0_attributes_gender_value', u'facepp_img_results_ES_faces_0_attributes_headpose_pitch_angle', u'facepp_img_results_ES_faces_0_attributes_headpose_roll_angle', u'facepp_img_results_ES_faces_0_attributes_headpose_yaw_angle', u'facepp_img_results_ES_faces_0_attributes_skinstatus_dark_circle', u'facepp_img_results_ES_faces_0_attributes_skinstatus_health', u'facepp_img_results_ES_faces_0_attributes_skinstatus_stain', u'facepp_img_results_ES_faces_0_attributes_smile_value', u'facepp_img_results_ES_faces_0_face_rectangle_height', u'facepp_img_results_ES_faces_0_face_rectangle_left', u'facepp_img_results_ES_faces_0_face_rectangle_top', u'facepp_img_results_ES_faces_0_face_rectangle_width', u'Human.Judge.for.Gender', u'Age_text_cat', u'Human.judgement.for.age_cat', u'facepp_img_results_ES_faces_0_attributes_age_value_cat'])
#train_data.drop(columns= ["Gender","profile_imageFeatures.BlueChannelMean","profile_imageFeatures.GrayScaleMean","profile_imageFeatures.GreenChannelMean","profile_imageFeatures.colorfulness","profile_imageFeatures.contrast","profile_imageFeatures.hueChannelMean","profile_imageFeatures.hueChannelVAR","profile_imageFeatures.naturalness","profile_imageFeatures.saturationChannelMean","profile_imageFeatures.saturationChannelVAR","profile_imageFeatures.sharpness","tweet_imageFeatures.BlueChannelMean","tweet_imageFeatures.GrayScaleMean","tweet_imageFeatures.GreenChannelMean","tweet_imageFeatures.RedChannelMean","tweet_imageFeatures.averageRGB","tweet_imageFeatures.brightness","tweet_imageFeatures.colorfulness","tweet_imageFeatures.contrast","tweet_imageFeatures.hueChannelMean","tweet_imageFeatures.hueChannelVAR","tweet_imageFeatures.naturalness","tweet_imageFeatures.saturationChannelMean","tweet_imageFeatures.saturationChannelVAR","tweet_imageFeatures.sharpness","Analytic","Authentic","Clout","Dic","Sixltr","Tone","WC","achieve","adj","adverb","affect","affiliation","anger","anx","article","assent","auxverb","bio","body","cause","certain","cogproc","compare","conj","death","differ","discrep","drives","family","feel","female","focusfuture","focuspast","focuspresent","friend","function.","health","hear","home","i","informal","ingest","insight","interrog","ipron","leisure","male","money","motion","negate","negemo","netspeak","nonflu","number","percept","posemo","power","ppron","prep","pronoun","quant","relativ","relig","reward","risk","sad","see","sexual","shehe","social","space","swear","tentat","they","time","verb","we","work","you","Annotation","profile_background_color","profile_sidebar_border_color","profile_text_color","profile_sidebar_fill_color","description","profile_link_color","lev_distance","description_sentiment_polarity","description_sentiment_subjectivity","combined_tweets","combined_tweets_polarity","combined_tweets_subjectivity","pigeo_results_lat","avg_tweet_polarity","face_pp_type","facepp_img_results_ES_faces_0_attributes_blur_blurness_value","facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_value","facepp_img_results_ES_faces_0_attributes_blur_motionblur_value","facepp_img_results_ES_faces_0_attributes_emotion_anger","facepp_img_results_ES_faces_0_attributes_emotion_disgust","facepp_img_results_ES_faces_0_attributes_emotion_fear","facepp_img_results_ES_faces_0_attributes_emotion_happiness","facepp_img_results_ES_faces_0_attributes_emotion_neutral","facepp_img_results_ES_faces_0_attributes_emotion_sadness","facepp_img_results_ES_faces_0_attributes_emotion_surprise","facepp_img_results_ES_faces_0_attributes_ethnicity_value","facepp_img_results_ES_faces_0_attributes_eyestatus_left_eye_status_no_glass_eye_open","facepp_img_results_ES_faces_0_attributes_eyestatus_right_eye_status_no_glass_eye_open","facepp_img_results_ES_faces_0_attributes_facequality_value","facepp_img_results_ES_faces_0_attributes_gender_value","facepp_img_results_ES_faces_0_attributes_headpose_pitch_angle","facepp_img_results_ES_faces_0_attributes_headpose_roll_angle","facepp_img_results_ES_faces_0_attributes_headpose_yaw_angle","facepp_img_results_ES_faces_0_attributes_skinstatus_dark_circle","facepp_img_results_ES_faces_0_attributes_skinstatus_health","facepp_img_results_ES_faces_0_attributes_skinstatus_stain","facepp_img_results_ES_faces_0_attributes_smile_value","facepp_img_results_ES_faces_0_face_rectangle_height","facepp_img_results_ES_faces_0_face_rectangle_left","facepp_img_results_ES_faces_0_face_rectangle_top","facepp_img_results_ES_faces_0_face_rectangle_width","Human.Judge.for.Gender","Age_text_cat","Human.judgement.for.age_cat","facepp_img_results_ES_faces_0_attributes_age_value_cat"])
# sys.exit()
#train_data = train_data[['Annotation','merged_tweets_description','avg_tweet_subjectivity','avg_tweet_favorite_count','avg_tweet_retweet_count','favourites_count','friends_count','statuses_count','followers_count', u'Analytic', u'Authentic', u'Clout', u'Dic', u'Sixltr', u'Tone', u'WC', u'achieve', u'adj', u'adverb', u'affect', u'affiliation', u'anger', u'anx', u'article', u'assent', u'auxverb', u'bio', u'body', u'cause', u'certain', u'cogproc', u'compare', u'conj', u'death', u'differ', u'discrep', u'drives', u'family', u'feel', u'female', u'focusfuture', u'focuspast', u'focuspresent', u'friend', u'function.', u'health', u'hear', u'home', u'i', u'informal', u'ingest', u'insight', u'interrog', u'ipron', u'leisure', u'male', u'money', u'motion', u'negate', u'negemo', u'netspeak', u'nonflu', u'number', u'percept', u'posemo', u'power', u'ppron', u'prep', u'pronoun', u'quant', u'relativ', u'relig', u'reward', u'risk', u'sad', u'see', u'sexual', u'shehe', u'social', u'space', u'swear', u'tentat', u'they', u'time', u'verb', u'we', u'work', u'you']]

train_data = train_data[['Annotation','Gender','Age_text_cat','merged_tweets_description', u'avg_tweet_subjectivity', u'Analytic', u'Authentic', u'Clout', u'Dic', u'Sixltr', u'Tone', u'WC', u'achieve', u'adj', u'adverb', u'affect', u'affiliation', u'anger', u'anx', u'article', u'assent', u'auxverb', u'bio', u'body', u'cause', u'certain', u'cogproc', u'compare', u'conj', u'death', u'differ', u'discrep', u'drives', u'family', u'feel', u'female', u'focusfuture', u'focuspast', u'focuspresent', u'friend', u'function.', u'health', u'hear', u'home', u'i', u'informal', u'ingest', u'insight', u'interrog', u'ipron', u'leisure', u'male', u'money', u'motion', u'negate', u'negemo', u'netspeak', u'nonflu', u'number', u'percept', u'posemo', u'power', u'ppron', u'prep', u'pronoun', u'quant', u'relativ', u'relig', u'reward', u'risk', u'sad', u'see', u'sexual', u'shehe', u'social', u'space', u'swear', u'tentat', u'they', u'time', u'verb', u'we', u'work', u'you']]
#print train_data.head()
train_data.to_csv("/mnt/volume/data/amir_fusing_data/Image_analysis_depressin_datasets/Image_analysis_depressin_datasets/trained_17.csv", sep = "\t")

print "File is ready!!!"