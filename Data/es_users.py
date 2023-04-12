from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from elasticsearch_dsl.connections import connections
es = Elasticsearch(['localhost:9201'], timeout= 160, max_retries=15, retry_on_timeout=True)
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

def delete(index, type, id):
    res =  es.delete(index=index,doc_type=type,id=id)
    print res
    return res

def getAllUsers():
    res = es.search(size = 1000, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={"query": {  "match_all": {}},"_source": ["screen_name"]}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllPhotosForUsers(screen_name):
    res = es.search(size = 1000, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query":{"bool":{"must":[{"match":{"extended_entities.media.type":"photo"}},{"match":{"user.screen_name":screen_name}}]}},"_source": ["screen_name", "lang", "extended_entities"]}, request_timeout=60)
    if len(res['hits']['hits']) == 0:
        screen_name = getUserProfileByID(screen_name)[0]["_source"]["screen_name"]
        res = es.search(size = 1000, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query":{"bool":{"must":[{"match":{"extended_entities.media.type":"photo"}},{"match":{"user.screen_name":screen_name}}]}},"_source": ["screen_name", "lang", "extended_entities"]}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllPhotosForUsers_Minus_Retweets(screen_name):
    res = es.search(size = 1000, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query":{"bool":{"must":[{"match":{"extended_entities.media.type":"photo"}},{"match":{"user.screen_name":screen_name}}], "must_not": [{"exists" : { "field" : "retweeted_status" }}]}},"_source": ["screen_name", "lang", "extended_entities"]}, request_timeout=60)
    if len(res['hits']['hits']) == 0:
        screen_name = getUserProfileByID(screen_name)[0]["_source"]["screen_name"]
        res = es.search(size = 1000, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query":{"bool":{"must":[{"match":{"extended_entities.media.type":"photo"}},{"match":{"user.screen_name":screen_name}}], "must_not": [{"exists" : { "field" : "retweeted_status" }}]}},"_source": ["screen_name", "lang", "extended_entities"]}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllStatusCoordinateUsers(anno_yes_list, must_not_match_these_users):
    res = es.search(size = 1000, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={ "query": { "bool": { "must": [ { "match": { "Annotation": "yes" } }, { "exists": { "field": "status.coordinates" } }, {"match": { "status.coordinates.type": "Point" }},{ "ids": { "values": anno_yes_list } } ], "must_not": [  {"ids" : { "values" : must_not_match_these_users }}] } } ,"_source": ["status"]}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllUsers_WithTweets_WithCoordinates(anno_yes_list, must_not_match_these_users):
	res = es.search(size = 10000, index="tweets", doc_type="goldstandard", body={ "query": { "bool": { "must": [ { "match": { "lang": "en" } }, { "exists": { "field": "coordinates.coordinates" } }, {"match" : { "Annotation" : "yes" }} ], "must_not": [ {"terms" : { "user.screen_name.keyword" : must_not_match_these_users }} ] } } , "aggs" : { "screen_name" : { "terms" : { "field" : "user.screen_name.keyword"}, "aggregations": { "hits": { "top_hits": { "size": 1 } } } } } }, request_timeout=60)
	all_users = res["aggregations"]['screen_name']['buckets']
	return all_users

def getAllUsers_WithTweets_WithPlaces_City(must_not_match_these_users):
	res = es.search(size = 10000, index="tweets", doc_type="goldstandard", body={ "query": { "bool": { "must": [ { "match": { "lang": "en" } }, { "match": { "place.place_type": "city" } } ], "must_not": [ { "terms": { "user.screen_name.keyword": must_not_match_these_users } } ] } }, "aggs": { "screen_name": { "terms": { "field": "user.screen_name.keyword" }, "aggregations": { "hits": { "top_hits": { "size": 1 } } } } }}, request_timeout=60)
	all_users = res["aggregations"]['screen_name']['buckets']
	return all_users

def getAllUsers_With_Profile_Location(must_not_match_these_users):
    res = es.search(size = 1000, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={ "query": { "bool": { "must": [ { "match": { "Annotation": "yes" } }, { "exists": { "field": "profile_location" } } ], "must_not": [ {"ids" : { "values" : must_not_match_these_users }} ] }}, "_source": ["screen_name","location","profile_location", "pigeo_results"] }, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    #print all_users
    return all_users

def getAllUsers_Without_Any_Location_Indication(must_not_match_these_users):
    res = es.search(size = 1000, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={ "query": { "bool": { "must": [ { "match": { "Annotation": "yes" } }, { "exists": { "field": "pigeo_results" } }], "must_not": [ {"ids" : { "values" : must_not_match_these_users }} ] } }, "_source" : ["screen_name", "pigeo_results"] }, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllUserProfiles():
    res = es.search(size = 1000, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={"query": {  "match_all": {}}}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllYesProfiles():
    res = es.search(size = 100, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={ "query": { "bool": { "must": [ { "match": { "Annotation": "yes" } }, { "exists": { "field": "avg_tweet_favorite_count" } } ] } }, "_source": [ "face_pp_type", "profile_background_color", "profile_sidebar_border_color", "profile_text_color", "profile_sidebar_fill_color", "followers_count", "listed_count", "description", "statuses_count", "friends_count", "profile_link_color", "favourites_count", "screen_name", "lev_distance", "description_sentiment_polarity", "description_sentiment_subjectivity", "combined_tweets_polarity", "combined_tweets_subjectivity", "pigeo_results.country", "pigeo_results.city", "pigeo_results.lon", "pigeo_results.state", "pigeo_results.lat", "Annotation", "avg_tweet_polarity", "avg_tweet_subjectivity", "avg_tweet_favorite_count", "avg_tweet_retweet_count", "profile_imageFeatures", "profile_OCR", "combined_tweets", "facepp_img_results" ]}, request_timeout=1560)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
#     return all_users
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllTweets():
    res = es.search(size = 1000, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query": {  "match_all": {}}, "_source" : ["text"]}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getUserProfile(screen_name):
    res = es.search(index="user_profiles", doc_type="goldstandard", body={"query": { "match": {"screen_name" : screen_name}}}, request_timeout=60)
    #print res
    user = res['hits']["hits"]
    return user

def getUserProfileByID(screen_name):
    res = es.get(index="user_profiles", doc_type="goldstandard", id=screen_name, request_timeout=60)
    return res

def getUser(screen_name):
    res = es.search(index="user_profiles", doc_type="goldstandard", body={"query": { "match": {"_id" : screen_name}}}, request_timeout=160)
    #print res
    profile_count = res['hits']['total']
    return profile_count

def getFollower(screen_name):
    res = es.search(index="user_profiles", doc_type="followers", body={"query": { "match": {"_id" : screen_name}}}, request_timeout=160)
    #print res
    profile_count = res['hits']['total']
    return profile_count

def getFriends(screen_name):
    res = es.search(index="user_profiles", doc_type="friends", body={"query": { "match": {"_id" : screen_name}}}, request_timeout=160)
    #print res
    profile_count = res['hits']['total']
    return profile_count

def getUserStoredInfo(screen_name):
    res = es.search(index="user_profiles", doc_type="goldstandard", body={"query": { "match": {"_id" : screen_name}}}, request_timeout=160)
    #print res
    profile_count = res['hits']['hits']
    return profile_count

def getFollowersCount(screen_name):
    res = es.search(index="user_profiles", doc_type="goldstandard", body={"_source" : "followers_count", "query": { "match": {"screen_name": screen_name}}}, request_timeout=160)
    followers_count = res['hits']['hits'][0]["_source"]["followers_count"]
    return followers_count

def getStoredTweetCount(screen_name):
    res = es.search(index="tweets", doc_type="goldstandard", body={"query": { "match": {"user.screen_name": screen_name}}}, request_timeout=160)
    #print res
    tweet_count = res['hits']['total']
    return tweet_count

def getStoredTweets(screen_name):
    res = es.search(size = 100, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query": { "match": {"user.screen_name": screen_name}}}, request_timeout=160)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_tweets = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_tweets = all_tweets + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_tweets

def getMax_And_Min_Id(screen_name):
    res = es.search(size = 1000, scroll = '1m', index="tweets", doc_type="goldstandard", body={"query": { "match": {"user.screen_name": screen_name}}}, request_timeout=160)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_tweets = res['hits']['hits']
    tweet_ids =[]
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_tweets = all_tweets + tweets
        #print "scroll_size: " + str(scroll_size)
    for doc in all_tweets:
        #print("%s) %s" % (doc['_id'], doc['_source']))
        id_str = (doc['_id'], doc['_source']['id_str'])
        #print ((int(id_str[1])))
        tweet_ids.append((int(id_str[1])))
    #print "Max", max(tweet_ids)
    #print tweet_ids
    # print min(tweet_ids)
    # print max(tweet_ids)
    return max(tweet_ids), min(tweet_ids)

def getTweet(id_str):
    res = es.search(index = "tweets", doc_type = "goldstandard", body = {"query": { "match": {"id_str":id_str} } }, request_timeout=160)
    tweet_count = res['hits']['total']
    return tweet_count

def getTweetContent(id_str):
    res = es.search(index = "tweets", doc_type = "goldstandard", body = {"query": { "match": {"id_str":id_str} } }, request_timeout=160)
    tweet_count = res['hits']['hits']
    return tweet_count

def indexTweet(tweet_json):
    try:
    	res = es.index(index = "tweets", doc_type = "goldstandard", id = tweet_json["id_str"], body = tweet_json)
        print res
    except:
        print sys.exc_info()[0]
    	return sys.exc_info()[0]
    print res

def indexUser(index_id, user_profile_json):
    res = es.index(index = "user_profiles", doc_type = "goldstandard", body = user_profile_json, id = index_id)
    print res

def indexFollower(index_id, user_profile_json):
    res = es.index(index = "user_profiles", doc_type = "followers" , body = user_profile_json, id = index_id)
    print res

def indexFriend(index_id, user_profile_json):
    res = es.index(index = "user_profiles", doc_type = "friends" , body = user_profile_json, id = index_id)
    print res

def get_All_Users_Without_Lev_Dis():
    res = es.search(size = 1000, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={"query": {"bool": {"must_not": {"exists": {"field": "lev_distance"}}}}}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def get_All_Users_Without_Factor(factor):
    res = es.search(size = 100, scroll = '1m', index="user_profiles", doc_type="goldstandard", body={"query": {"bool": {"must_not": {"exists": {"field": factor}}}}}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllLocUsers():
    res = es.search(size = 1000, scroll = '1m', index="location", doc_type="user", body={"query": {  "match_all": {}}}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users
#
def getAllLocUsersByAnno(anno):
    res = es.search(size = 1000, scroll = '1m', index="location", doc_type="user", body={"query": {  "match": {"Annotation": anno}}}, request_timeout=60)
    s_id = res['_scroll_id']
    scroll_size = res['hits']['total']
    #print res
    all_users = res['hits']['hits']
    while (scroll_size > 0):
        res = es.scroll(scroll_id = s_id, scroll = '1m' , request_timeout=160)
        s_id = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
        tweets = res['hits']['hits']
        all_users = all_users + tweets
        #print "scroll_size: " + str(scroll_size)
    return all_users

def getAllLocUsersByType():
    res = es.search(size = 1000, scroll = '1m', index="location", doc_type="user", body={ "query": { "match_all": {} }, "aggs": { "loc_type": { "terms": { "field": "Type.keyword", "size": 10000 }, "aggregations": { "hits": { "top_hits": { "size": 10000 } } } } } }, request_timeout=60)
    all_users = res["aggregations"]['loc_type']['buckets']
    return all_users


def update_Loc_Field(index_id, field, value):
    res = es.update(index = "location", doc_type = "user" , body={"doc" : {"tags" : [ "updated" ], field: value }}, id = index_id)

def index_Loc_Field(index_id, loc_json):
    res = es.index(index = "location", doc_type = "user" , body=loc_json, id = index_id)

def update_gs_Field(index_id, field, value):
    res = es.update(index = "user_profiles", doc_type = "goldstandard" , body={"doc" : {"tags" : [ "updated" ], field: value }}, id = index_id)

def update_Tweet_Field(index_id, field, value):
    res = es.update(index = "tweets", doc_type = "goldstandard" , body={"doc" : {"tags" : [ "updated" ], field: value }}, id = index_id)

def update_gs_followee_count(index_id, count):
    res = es.update(index = "user_profiles", doc_type = "followers" , body={"doc" : {"tags" : [ "updated" ],"gs_followee_count": count }}, id = index_id)

def get_gs_followee_count(index_id):
    res = es.search(index = "user_profiles", doc_type = "followers" , body={"query": { "match": {"_id":index_id} } , "_source": ["gs_followee_count"]})
    return res["hits"]["hits"][0]["_source"]["gs_followee_count"]

def get_gs_friends_count(index_id):
    res = es.search(index = "user_profiles", doc_type = "friends" , body={"query": { "match": {"_id":index_id} } , "_source": ["gs_friends_count"]})
    return res["hits"]["hits"][0]["_source"]["gs_friends_count"]

def update_gs_friends_count(index_id, count):
    res = es.update(index = "user_profiles", doc_type = "friends" , body={"doc" : {"tags" : [ "updated" ],"gs_friends_count": count }}, id = index_id)

def update_lev(index_id, dis):
    res = es.update(index = "user_profiles", doc_type = "goldstandard" , body={"doc" : {"tags" : [ "updated" ],"lev_distance": dis }}, id = index_id)

def update_desc_senti(index_id, polarity, subjectivity):
    res = es.update(index = "user_profiles", doc_type = "goldstandard" , body={"doc" : {"tags" : [ "updated" ],"description_sentiment_subjectivity": subjectivity, "description_sentiment_polarity": polarity }}, id = index_id)
