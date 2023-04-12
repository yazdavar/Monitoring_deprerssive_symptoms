library (ggplot2)
library(caTools)
library(Amelia)
library(ggplot2)
library(VIM)
library(mice)
library(caret)
library(AppliedPredictiveModeling)
library(Hmisc)
library(RANN)
library(randomForest)
library(Boruta)
library(FSelector)
library(mlr)
library(corrplot)
library(missMDA)
library(tidyr)
library (dplyr)
library (broom)
library(DMwR)
library(corrplot)
library(PerformanceAnalytics)
library(heuristica)
library(irr)
library(lpSolve)
library(ggfortify)
library(ggpubr)
library("cowplot")
library(magrittr)
library("ggpubr")

#df <- read.csv("/home/amir/work/code/Depression_image_analysis/Mohammad_DataFrame.csv", header = TRUE,sep="," )
#df <- read.csv("/home/amir/work/code/Depression_image_analysis/df_filtered_textual.csv", header = TRUE,sep=",", na.strings =c("")  )

df <- read.csv("/media/amir/Amir/Image_analysis_depressin_datasets/combined_DF_annotated_age.tsv", header = TRUE,sep="\t", na.strings =c("")  )
#check the the status of missing values
names(df)
head(df)
sort(sapply(df,function(x) sum(is.na(x))))
missmap(df, main = "Missing values vs observed")

#114,129,121
names (df)

#sum(is.na(df$Human.judgement.for.age))

sum()


#accroding to my observation and the previous chart I decided to drop several of Independent variables.
#drop,4,5,32,33, these columns are including id_listing_anon, id_user_anon, p2_p3_click_through_score, p3_inquiry_score, days_since_last_booking, Price_booked_most_recent, occ_occupancy_plus_minus_7_ds_night,occ_occupancy_plus_minus_14_ds_night
#str(df)

#df_goonmeet <- df[,c(1,115)]
#summary(df_goonmeet)

#exclude column 130: tweets, 19: media_id_str, 122: description, 189: facepp_img_results_ES_time_used, 188:facepp_img_results_ES_request_id,
#187:facepp_img_results_ES_faces_0_face_token 

df <- df[, -c(1,2,19,130,122,187:189)]
names(df)

#df <- df[, -c(2,19,122,187:189)]

#keep Ethnicity
df <- df[, -c(2,19,130,122,187:189)] 

#str(df)

write.csv(df, file = "/home/amir/work/code/Depression_image_analysis/df_filtered_textual_annotated_age.csv", fileEncoding = "UTF-8")

names(df)
#drawing percentage chart
mice_plot <- aggr(df, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(df), cex.axis=.3,
                  gap=3, ylab=c("Missing data","Pattern"))


str(df)
#for my convinent and acessability to 
#reordering to make dealing easier with factor variables

fact_list= c("profile_link_color","profile_background_color","profile_sidebar_border_color"," profile_text_color","profile_sidebar_fill_color",
             "pigeo_results_country","pigeo_results_city","pigeo_results_state","face_pp_type", "facepp_img_results_ES_faces_0_attributes_ethnicity_value"," facepp_img_results_ES_faces_0_attributes_gender_value","facepp_img_results_ES_faces_0_attributes_glass_value","Annotation")


#num_list =
for (item in fact_list) {
  v <- (grep(item, colnames(df)))
  vector <- c(vector, v)
}
#112,113,114,116,128,129,131,137,*138,152,168
names(df)

#1_annotation,2_age_text, 3_gender_text,
#137_face_pp_type,138_age_from_image, 167_gender_image
df_reordered <- df[,c(112,1,2,137,138,183,184,185,167,3:111,115,117:127,130,132:136,139:151,153:166,169:182,113,114,116,128,129,131,152,168)]
str(df_reordered)
names(df_reordered)

head(df_reordered$X)

names(df_reordered)
write.csv(df_reordered, file = "/home/amir/work/code/Depression_image_analysis/df_reordered_annotated_age.csv", fileEncoding = "UTF-8")
str(df)

sum(is.na(df_reordered$Annotation))
summary(df_reordered$Annotation)


df_reordered$Annotation<-as.factor(df_reordered$Annotation)
 
df_reordered$Human.Judge.for.Gender<-as.factor(df_reordered$Human.Judge.for.Gender)
df_reordered$Description..How.you.come.up.with.the.age.<-as.factor(df_reordered$Description..How.you.come.up.with.the.age.)
df_reordered$pigeo_results_country<-as.factor(df_reordered$pigeo_results_country)
df_reordered$pigeo_results_state<-as.factor(df_reordered$pigeo_results_state)
df_reordered$pigeo_results_city<-as.factor(df_reordered$pigeo_results_city)
df_reordered$face_pp_type<-as.factor(df_reordered$face_pp_type)
df_reordered$facepp_img_results_ES_faces_0_attributes_gender_value<-as.factor(df_reordered$facepp_img_results_ES_faces_0_attributes_gender_value)
df_reordered$profile_link_color<-as.factor(df_reordered$profile_link_color)
df_reordered$profile_background_color<-as.factor(df_reordered$profile_background_color)
df_reordered$profile_sidebar_border_color<-as.factor(df_reordered$profile_sidebar_border_color)
df_reordered$profile_sidebar_fill_color<-as.factor(df_reordered$profile_sidebar_fill_color)
df_reordered$facepp_img_results_ES_faces_0_attributes_ethnicity_value<-as.factor(df_reordered$facepp_img_results_ES_faces_0_attributes_ethnicity_value)
df_reordered$facepp_img_results_ES_faces_0_attributes_glass_value<-as.factor(df_reordered$facepp_img_results_ES_faces_0_attributes_glass_value)
df_reordered$Gender <- as.numeric(as.character(df_reordered$Gender))
df_reordered$female <- as.numeric(as.character(df_reordered$female))
df_reordered$feel <- as.numeric(as.character(df_reordered$feel))
df_reordered$drives <- as.numeric(as.character(df_reordered$drives))
df_reordered$death <- as.numeric(as.character(df_reordered$death))
df_reordered$affect <- as.numeric(as.character(df_reordered$affect))
df_reordered$profile_imageFeatures.BlueChannelMean <- as.numeric(as.character(df_reordered$profile_imageFeatures.BlueChannelMean))
df_reordered$profile_imageFeatures.GreenChannelMean <- as.numeric(as.character(df_reordered$profile_imageFeatures.GreenChannelMean))
df_reordered$profile_imageFeatures.averageRGB <- as.numeric(as.character(df_reordered$profile_imageFeatures.averageRGB))
df_reordered$profile_imageFeatures.hueChannelMean <- as.numeric(as.character(df_reordered$profile_imageFeatures.hueChannelMean))
df_reordered$tweet_imageFeatures.hueChannelMean <- as.numeric(as.character(df_reordered$tweet_imageFeatures.hueChannelMean))



#df_reordered <-rename(df_reordered, Class = Annotation )
summary(df_reordered$Annotation)

str(df_reordered)

df_reordered$combined_tweets_polarity <- sapply(df_reordered$combined_tweets_polarity, function(x) {-1*x} )

str(df_reordered)
names (df_reordered)
#subset(df_reordered, !is.na(df_reordered$Annotation))
df_reordered_remove_50S = subset(df_reordered, select = -c(facepp_img_results_ES_faces_0_attributes_blur_blurness_threshold,facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_threshold,facepp_img_results_ES_faces_0_attributes_blur_motionblur_threshold,facepp_img_results_ES_faces_0_attributes_facequality_threshold,facepp_img_results_ES_faces_0_attributes_smile_threshold,facepp_img_results_ES_faces_0_attributes_glass_value))


#remove NAN from the Annotation class
df_reordered_remove_50S <-  df_reordered_remove_50S[!is.na(df_reordered_remove_50S$Annotation),]
df_reordered_remove_50S$Annotation <- sapply(df_reordered_remove_50S$Annotation, function(x) if (x == 'no' ) {"Control"} else {"Depressed"})

summary (df_reordered_remove_50S$Annotation)
 df_reordered_remove_50S$Annotation
 
#df_reordered_omitted_nans_filtered_50s_factor <- df_reordered_omitted_nans_filtered_50s [,sapply(df_reordered_omitted_nans_filtered_50s, function (x) class (x)== 'factor')]
 mice_plot <- aggr(df_reordered_remove_50S, col=c('navyblue','yellow'),
                   numbers=TRUE, sortVars=TRUE,
                   labels=names(df), cex.axis=.3,
                   gap=3, ylab=c("Missing data","Pattern"))

#------------------------------Handling colors--------------------------
df_reordered_remove_50S$profile_sidebar_fill_color <- paste("#", df_reordered_remove_50S$profile_sidebar_fill_color, sep="")
rgb_mat <- col2rgb(df_reordered_remove_50S$profile_sidebar_fill_color)
rgb_mat <- t(rgb_mat)
rgb_mat_df <- as.data.frame(rgb_mat)
df_reordered_remove_50S$profile_sidebar_fill_color_red<-rgb_mat_df$red
df_reordered_remove_50S$profile_sidebar_fill_color_green<-rgb_mat_df$green
df_reordered_remove_50S$profile_sidebar_fill_color_blue<-rgb_mat_df$blue

#------------------------------------------------------

df_reordered_remove_50S$profile_sidebar_border_color <- paste("#", df_reordered_remove_50S$profile_sidebar_border_color, sep="")
rgb_mat <- col2rgb(df_reordered_remove_50S$profile_sidebar_border_color)
rgb_mat <- t(rgb_mat)
rgb_mat_df <- as.data.frame(rgb_mat)
df_reordered_remove_50S$profile_sidebar_border_color_red<-rgb_mat_df$red
df_reordered_remove_50S$profile_sidebar_border_color_green<-rgb_mat_df$green
df_reordered_remove_50S$profile_sidebar_border_color_blue<-rgb_mat_df$blue

#------------------------------------------
df_reordered_remove_50S$profile_background_color <- paste("#", df_reordered_remove_50S$profile_background_color, sep="")
rgb_mat <- col2rgb(df_reordered_remove_50S$profile_background_color)
rgb_mat <- t(rgb_mat)
rgb_mat_df <- as.data.frame(rgb_mat)
df_reordered_remove_50S$profile_background_color_red<-rgb_mat_df$red
df_reordered_remove_50S$profile_background_color_green<-rgb_mat_df$green
df_reordered_remove_50S$profile_background_color_blue<-rgb_mat_df$blue
#-------------------------------------------------
df_reordered_remove_50S$profile_link_color <- paste("#", df_reordered_remove_50S$profile_link_color, sep="")
rgb_mat <- col2rgb(df_reordered_remove_50S$profile_link_color)
rgb_mat <- t(rgb_mat)
rgb_mat_df <- as.data.frame(rgb_mat)
df_reordered_remove_50S$profile_link_color_red<-rgb_mat_df$red
df_reordered_remove_50S$profile_link_color_green<-rgb_mat_df$green
df_reordered_remove_50S$profile_link_color_blue<-rgb_mat_df$blue


#---------------------------------------------------------
df_reordered_remove_50S$profile_text_color <- paste("#", df_reordered_remove_50S$profile_text_color, sep="")
rgb_mat <- col2rgb(df_reordered_remove_50S$profile_text_color)
rgb_mat <- t(rgb_mat)
rgb_mat_df <- as.data.frame(rgb_mat)
df_reordered_remove_50S$profile_text_color_red<-rgb_mat_df$red
df_reordered_remove_50S$profile_text_color_green<-rgb_mat_df$green
df_reordered_remove_50S$profile_text_color_blue<-rgb_mat_df$blue

df_reordered_color_handled = subset(df_reordered_remove_50S, select = -c(profile_link_color,profile_text_color,profile_background_color,profile_sidebar_border_color,profile_sidebar_fill_color))
str(df_reordered_color_handled)

#-------Remove Pigeo Factors----------------------
df_reordered_location_handled = subset(df_reordered_color_handled, select = -c(pigeo_results_country,pigeo_results_state,pigeo_results_city))

#str(df_reordered_location_handled$facepp_img_results_ES_faces_0_attributes_ethnicity_value)
#df_reordered_location_handled$facepp_img_results_ES_faces_0_attributes_ethnicity_value
#df_reordered_location_handled$facepp_img_results_ES_faces_0_attributes_ethnicity_value


#selecting only visual features
df_tweet_imageFeatures <- select(df_reordered_location_handled,Annotation,starts_with("tweet_imageFeatures"))
df_facepp_img_results <- select(df_reordered_location_handled,starts_with("facepp_img_results"))
df_profile <- select(df_reordered_location_handled,starts_with("profile_"))

merged_tweet_imageFeatures <- cbind(df_tweet_imageFeatures,df_facepp_img_results,df_profile )
merged_tweet_imageFeatures[] <- lapply(merged_tweet_imageFeatures, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
sum(is.na(merged_tweet_imageFeatures))
summary(merged_tweet_imageFeatures)
names(merged_tweet_imageFeatures)
write.csv(merged_tweet_imageFeatures, file = "/home/amir/work/code/Depression_image_analysis/merged_tweet_imageFeatures.csv", fileEncoding = "UTF-8")
#merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, Class = Annotation )

#----------------------Done Preprocessing------------------------------




names(df_reordered_location_handled)
#df_reordered_loc_anotated <- group_by(df_reordered_location_handled, df_reordered_location_handled$Annotation)
df_reordered_location_handled$Annotation

yes_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "Depressed")
no_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "Control")


summarize(df_reordered_loc_anotated, age_text = mean(Age, na.rm = TRUE), gender_text =mean(Gender, na.rm = TRUE) , age_image= mean(facepp_img_results_ES_faces_0_attributes_age_value, na.rm = TRUE), face_pp_type)
summarize(df_reordered_loc_anotated, face_pp_type)

#Getting Chi-squared value for image extraction
#remove Nans
df_face_pp_not_nan <-  df_reordered_location_handled[!is.na(df_reordered_location_handled$face_pp_type),]
df_face_pp_not_nan_subset <- subset(df_face_pp_not_nan, face_pp_type == 'Face_NOT_found_in_profile_or_in_media')
df_face_pp_not_vs_annotation <- subset(df_face_pp_not_nan_subset, select = c(face_pp_type,Annotation))
table(df_face_pp_not_vs_annotation)
df_face_pp_not_vs_annotation <- table(df_anotaion_selecterd_2$face_pp_type, df_anotaion_selecterd_2$Annotation )
#remove levels
df_face_pp_not_vs_annotation <- table(droplevels(df_anotaion_selecterd_2$face_pp_type),df_anotaion_selecterd_2$Annotation)
df_face_pp_not_vs_annotation

if (chisq.test(df_face_pp_not_vs_annotation)$p.value < 0.001){
  print ('***')
}else if (chisq.test(df_face_pp_not_vs_annotation)$p.value < 0.01){
  print ('**')
} else if (chisq.test(df_face_pp_not_vs_annotation)$p.value < 0.05){
  print ('*')
}
chisq.test(df_face_pp_not_vs_annotation)



names(yes_df)

#-------log_transformed----------------
knnOutput_yes_logged <- log(knnOutput_yes)
preProc = preProcess(knnOutput_yes_logged, "medianImpute")
knnOutput_yes_logged_imputed <- predict(preProc, knnOutput_yes_logged)

#Remove infinate
knnOutput_yes_logged_imputed_inf_handled <- knnOutput_yes_logged_imputed[!is.infinite(rowSums(knnOutput_yes_logged_imputed)),]


#-------------Sentiment analysis-----------------------------




#----------------------------Overal has face features -------------------------
yes_df_has_face_feature <- filter(yes_df, face_pp_type == 'Face_found_in_media' | face_pp_type == "Face_found_in_profile")
#yes_df_has_face_feature <- filter(yes_df, face_pp_type == 'Face_found_in_media')
#Face_found_in_media, Face_found_in_profile, Face_NOT_found_in_profile_or_in_media
no_df_has_face_feature <- filter(no_df, face_pp_type == 'Face_found_in_media' | face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == 'Face_found_in_media' )

yes_subjectivity <- select(yes_df_has_face_feature, facepp_img_results_ES_faces_0_attributes_emotion_anger, anger ,facepp_img_results_ES_faces_0_attributes_emotion_disgust, facepp_img_results_ES_faces_0_attributes_emotion_happiness, sad,facepp_img_results_ES_faces_0_attributes_emotion_sadness,negemo, posemo,description_sentiment_polarity,combined_tweets_polarity,facepp_img_results_ES_faces_0_attributes_smile_value)
no_subjectivity <- select(no_df_has_face_feature, facepp_img_results_ES_faces_0_attributes_emotion_anger, anger ,facepp_img_results_ES_faces_0_attributes_emotion_disgust, facepp_img_results_ES_faces_0_attributes_emotion_happiness, sad,facepp_img_results_ES_faces_0_attributes_emotion_sadness,negemo, posemo,description_sentiment_polarity,combined_tweets_polarity,facepp_img_results_ES_faces_0_attributes_smile_value)


rename_yes_subjectivity <-rename(yes_subjectivity, Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Content_sadness = sad, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness, Content_negative_emotion= negemo, Content_positive_emotion =posemo, Description_sentiment= description_sentiment_polarity,Average_content_sentiment= combined_tweets_polarity,Face_smile= facepp_img_results_ES_faces_0_attributes_smile_value)
rename_no_subjectivity <-rename(no_subjectivity, Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Content_sadness = sad,Face_sadness = facepp_img_results_ES_faces_0_attributes_smile_value,  Content_negative_emotion= negemo, Content_positive_emotion =posemo, Description_sentiment= description_sentiment_polarity,Average_content_sentiment= combined_tweets_polarity,Face_smile= facepp_img_results_ES_faces_0_attributes_smile_value)

head(rename_no_subjectivity)
rename_yes_subjectivity <-rename(yes_subjectivity, Cont.Sadness = sad,Cont.Angriness = anger,Cont.Neg.Emot = negemo, Cont.Pos.Emot = posemo, Avg.Cont.Sent. =combined_tweets_polarity, Descrip.Sent. = description_sentiment_polarity , Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness, Face_smile=facepp_img_results_ES_faces_0_attributes_smile_value)

head(rename_yes_subjectivity)
knnOutput_yes <- knnImputation(rename_yes_subjectivity[, !names(rename_yes_subjectivity) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput_yes)

rename_no_subjectivity <-rename(no_subjectivity,Content_sadness = sad,Content_angriness = anger,Content_negaitive_emotion = negemo, Content_positive_emotion = posemo,Average_content_sentiment=combined_tweets_polarity, Description_sentiment = description_sentiment_polarity , Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness, Face_smile=facepp_img_results_ES_faces_0_attributes_smile_value)

rename_no_subjectivity <-rename(no_subjectivity, Cont.Sadness = sad,Cont.Angriness = anger,Cont.Neg.Emot = negemo, Cont.Pos.Emot = posemo, Avg.Cont.Sent. =combined_tweets_polarity, Descrip.Sent. = description_sentiment_polarity , Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness, Face_smile=facepp_img_results_ES_faces_0_attributes_smile_value)
head(rename_no_subjectivity)

knnOutput_no <- knnImputation(rename_no_subjectivity[, !names(rename_no_subjectivity) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput_no)
head(knnOutput_no)

M <- cor(knnOutput_yes)
M <- cor(knnOutput_no)

corrplot(M,order = "hclust",addrect = 2 ,method = "circle")
corrplot(M,order = "hclust",addrect = 2 ,method = "number") # Display the correlation coefficient
corrplot.mixed(M, number.cex = .7)
chart.Correlation(knnOutput_no, histogram=TRUE, pch=19)
res2 <- cor.mtest(knnOutput_no, conf.level = .95)
res2 <- cor.mtest(knnOutput_yes, conf.level = .90)

## specialized the insignificant value according to the significant level
#corrplot(knnOutput_yes,is.corr = FALSE  ,p.mat = res1$p, sig.level = .05)
corrplot(M ,p.mat = res2$p, sig.level = .05, order = "hclust",addrect = 2)

#tl.cex = 1
corrplot(M ,p.mat = res2$p, sig.level = .1, tl.cex = 0.8 , tl.col = 'black', tl.srt = 45)
corrplot(M ,p.mat = res2$p, sig.level = .1 , tl.col = 'black')
#corrplot(res2$r, type="upper", order="hclust", p.mat = res2$P, sig.level = 0.01, insig = "blank")
dev.off()
#-------------------------Gender----------------------------------------------------------------------
#facepp_img_results_ES_faces_0_attributes_ethnicity_value
yes_df_has_face_feature <- filter(yes_df, face_pp_type == 'Face_found_in_media' | face_pp_type == "Face_found_in_profile")
#yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_media")
#Face_found_in_media, Face_found_in_profile, Face_NOT_found_in_profile_or_in_media
#no_df_has_face_feature <- filter(no_df, face_pp_type == 'Face_found_in_media' | face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_profile")

yes_gender<- select(yes_df_has_face_feature, Gender,facepp_img_results_ES_faces_0_attributes_gender_value, female, male)
no_gender <- select(no_df_has_face_feature, Gender,facepp_img_results_ES_faces_0_attributes_gender_value, female, male)

yes_gender
no_gender

knnOutput_no <- knnImputation(no_gender[, !names(no_gender) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput_no)
knnOutput_no

yes_gender$facepp_img_results_ES_faces_0_attributes_gender_value <- sapply(yes_gender$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})
no_gender$facepp_img_results_ES_faces_0_attributes_gender_value <- sapply(no_gender$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})


#dim(no_gender)
sum(is.na(no_gender))
no_gender_nan_handled <- na.omit(no_gender)
sum(is.na(no_gender_nan_handled))
yes_gender$Gender <- sapply(yes_gender$Gender, function(x) if (x > 0 ) {0} else {1})
no_gender_nan_handled$Gender <- sapply(no_gender_nan_handled$Gender, function(x) if (x < 0 ) {1} else {0})
#rename_yes_subjectivity <-rename(yes_subjectivity, Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness)
#rename_no_subjectivity <-rename(no_subjectivity, Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness)
yes_gender
no_gender_nan_handled

summary(no_gender$Gender)
nrow(yes_gender)
yes_gender
knnOutput_yes <- knnImputation(yes_gender[, !names(yes_gender) %in% "medv"], k=1)  # perform knn imputation.
anyNA(knnOutput_yes)

knnOutput_no <- knnImputation(rename_no_subjectivity[, !names(rename_no_subjectivity) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput_no)

knnOutput_yes
M <- cor(knnOutput_yes)
M <- cor(knnOutput_no)

corrplot(M,order = "hclust",addrect = 2 ,method = "circle")
corrplot(M,order = "hclust",addrect = 2 ,method = "number") # Display the correlation coefficient
corrplot.mixed(M, number.cex = .7)
chart.Correlation(knnOutput_no, histogram=TRUE, pch=19)
res1 <- cor.mtest(knnOutput_no, conf.level = .95)
res2 <- cor.mtest(knnOutput_yes, conf.level = .99)

## specialized the insignificant value according to the significant level
#corrplot(knnOutput_yes,is.corr = FALSE  ,p.mat = res1$p, sig.level = .05)
corrplot(M ,p.mat = res1$p, sig.level = .05)






#-----------------------------------------------------------
# for non-depressed users
head(no_gender_nan_handled)
predictions <- no_gender_nan_handled[,"facepp_img_results_ES_faces_0_attributes_gender_value"]
actual <- no_gender_nan_handled[,"Gender"]
confusionMatrix(predictions, actual)

no_gender_nan_handled

#cohen.kappa
x=cbind(no_gender_nan_handled$facepp_img_results_ES_faces_0_attributes_gender_value,no_gender_nan_handled$Gender)
x
kappa2(x, "unweighted")

agree(x,tolerance=0)
#--------------------
#for depressed users
predictions <- yes_gender[,"facepp_img_results_ES_faces_0_attributes_gender_value"]
actual <- yes_gender[,"Gender"]
confusionMatrix(predictions, actual)

#cohen.kappa
x=cbind(yes_gender$facepp_img_results_ES_faces_0_attributes_gender_value,yes_gender$Gender)
x
kappa2(x, "unweighted")

#percentage aggrement
agree(x,tolerance=0)
# bhapkar(x)

#----------------------------------Age-------------------------------------
#yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_media"| face_pp_type == "Face_found_in_profile")
#yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_profile")
yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_media")
#Face_found_in_media, Face_found_in_profile, Face_NOT_found_in_profile_or_in_media
#no_df_has_face_feature <- filter(no_df, face_pp_type == 'Face_found_in_media' | face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_profile")

names(yes_df_has_face_feature)

yes_age<- select(yes_df_has_face_feature,friends_count ,followers_count,facepp_img_results_ES_faces_0_attributes_age_value,family,friend,Age,Gender,facepp_img_results_ES_faces_0_attributes_gender_value, female, male)
no_age <- select(no_df_has_face_feature,friends_count ,followers_count,facepp_img_results_ES_faces_0_attributes_age_value,family,friend,Age,Gender,facepp_img_results_ES_faces_0_attributes_gender_value, female, male)

yes_age
no_age


knnOutput_no <- knnImputation(no_age[, !names(no_age) %in% "medv"])  # perform knn imputation.
anyNA(no_age)
knnOutput_no

yes_age$facepp_img_results_ES_faces_0_attributes_gender_value <- sapply(yes_age$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})
no_age$facepp_img_results_ES_faces_0_attributes_gender_value <- sapply(no_age$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})


#dim(no_gender)
sum(is.na(yes_age))
yes_age_nan_handled <- na.omit(yes_age)
sum(is.na(yes_age_nan_handled))
yes_age$Gender <- sapply(yes_age$Gender, function(x) if (x > 0 ) {0} else {1})
no_gender_nan_handled$Gender <- sapply(no_gender_nan_handled$Gender, function(x) if (x < 0 ) {1} else {0})
#rename_yes_subjectivity <-rename(yes_subjectivity, Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness)
#rename_no_subjectivity <-rename(no_subjectivity, Face_anger = facepp_img_results_ES_faces_0_attributes_emotion_anger, Face_disgust = facepp_img_results_ES_faces_0_attributes_emotion_disgust,Face_happiness =facepp_img_results_ES_faces_0_attributes_emotion_happiness, Face_sadness = facepp_img_results_ES_faces_0_attributes_emotion_sadness)
names(yes_age)
no_gender_nan_handled

summary(no_gender$Gender)
nrow(yes_gender)
yes_gender
knnOutput_yes <- knnImputation(yes_gender[, !names(yes_gender) %in% "medv"], k=1)  # perform knn imputation.
anyNA(knnOutput_yes)

knnOutput_no <- knnImputation(rename_no_subjectivity[, !names(rename_no_subjectivity) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput_no)

str(yes_age_nan_handled)

x <- yes_age_nan_handled$facepp_img_results_ES_faces_0_attributes_age_value
y <- yes_age_nan_handled$Age
plot(x, y, main="Scatterplot Example", 
     xlab="Image_age_value", ylab="Content_Image", pch=19)
abline(lm(y~x), col="red")

rcorr(x,y, type="pearson")
ggplot(yes_age_nan_handled, aes(x=x, y=y, shape = yes_age_nan_handled$facepp_img_results_ES_faces_0_attributes_gender_value, color = yes_age_nan_handled$facepp_img_results_ES_faces_0_attributes_gender_value )) +
  geom_point(size=2, shape=23)+ geom_smooth(method=lm )

df <- data.frame(x,y)
dim(df)
M <- cor(df)

corrplot(M,order = "hclust",addrect = 2 ,method = "circle")
corrplot(M,order = "hclust",addrect = 2 ,method = "number")


#----------------Gold_standard_for_age---------------------------------
yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_media"| face_pp_type == "Face_found_in_profile")
yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_media"| face_pp_type == "Face_found_in_profile")


yes_df_nan_handled <-  yes_df[!is.na(yes_df$Human.judgement.for.age),]
yes_df_nan_handled <- filter(yes_df_nan_handled, yes_df_nan_handled$Human.judgement.for.age > 0)
yes_df_has_face_feature <-  yes_df[!is.na(yes_df_has_face_feature$Human.judgement.for.age),]
yes_df_has_face_feature <- filter(yes_df_has_face_feature, yes_df_has_face_feature$Human.judgement.for.age > 0)


summary(yes_df_nan_handled$Human.judgement.for.age)
dim(yes_df_nan_handled)

no_df_nan_handled <-  no_df[!is.na(no_df$Human.judgement.for.age),]
no_df_nan_handled <- filter(no_df_nan_handled, no_df_nan_handled$Human.judgement.for.age > 0)

mean(no_df_nan_handled$Human.judgement.for.age)
t.test(no_df_nan_handled$Human.judgement.for.age, yes_df_nan_handled$Human.judgement.for.age)

summary(yes_df_nan_handled$Human.judgement.for.age)
summary(no_df_nan_handled$Human.judgement.for.age)



yes_df_has_face_feature_removed_nan <-  yes_df_has_face_feature[!is.na(yes_df_has_face_feature$Description..How.you.come.up.with.the.age.),]
no_df_has_face_feature_removed_nan <-  no_df_has_face_feature[!is.na(no_df_has_face_feature$Description..How.you.come.up.with.the.age.),]


#converting continous data to factors
yes_df_has_face_feature_removed_nan_with_age <- filter(yes_df_has_face_feature_removed_nan, yes_df_has_face_feature_removed_nan$Human.judgement.for.age > 0)
summary(yes_df_has_face_feature_removed_nan_with_age$Human.judgement.for.age)

#create PDF from non-depressed and depressed users
yes_df_has_face_feature_removed_nan_with_age$age_text_cat_yes <-cut(yes_df_has_face_feature_removed_nan_with_age$Age, c(14,19,23,34,46,60))
no_df_has_face_feature_removed_nan$age_text_cat_no <-cut(no_df_has_face_feature_removed_nan$Age, c(14,19,23,34,46,60))
summary(yes_df_has_face_feature_removed_nan_with_age$age_text_cat)

#-----------------------------------------------------------------------------
#plotting PDF
deppressed <- data.frame(age_text_cat= yes_df_nan_handled$Human.judgement.for.age,
                    Annotation=yes_df_nan_handled$Annotation)
controled <- data.frame(age_text_cat= no_df_nan_handled$Human.judgement.for.age,
                     Annotation=no_df_nan_handled$Annotation)

merged_age_categories <- rbind(deppressed, controled)
Annotation <- as.factor(merged_age_categories$Annotation)

merged_age_categories$Annotation <- sapply(merged_age_categories$Annotation, function(x) if (x == 'no' ) {"Control"} else {"Depressed"})
merged_age_categories <-rename(merged_age_categories, Class = Annotation )
#PDF for depressed vs Non-depressed
summary(merged_age_categories)
p <- ggplot(merged_age_categories, aes(x=age_text_cat, colour=Class)) + geom_density()+ labs(x = "Age Distribution", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))+ 
  theme_classic()#+ theme(axis.text=element_text(size=12),
p + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15)
)                        #axis.title=element_text(size=14,face="bold"))


#----------------------------------------------------------------

yes_df_has_face_feature_removed_nan_with_age$age_text_cat <-cut(yes_df_has_face_feature_removed_nan_with_age$Age, c(14,19,23,34,46,60))
yes_df_has_face_feature_removed_nan_with_age$age_image_cat <-cut(yes_df_has_face_feature_removed_nan_with_age$facepp_img_results_ES_faces_0_attributes_age_value, c(14,19,23,34,46,60))
yes_df_has_face_feature_removed_nan_with_age$age_human_cat <-cut(yes_df_has_face_feature_removed_nan_with_age$Human.judgement.for.age, c(14,19,23,34,46,60))

#Human.Judge.for.Gender
#age_human_cat

names(yes_df_has_face_feature_removed_nan_with_age)
dim(yes_df_has_face_feature_removed_nan_with_age)

#getting frequency
count (yes_df_has_face_feature_removed_nan_with_age ,yes_df_has_face_feature_removed_nan_with_age$age_human_cat )

yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories <-  yes_df_has_face_feature_removed_nan_with_age[!is.na(yes_df_has_face_feature_removed_nan_with_age$age_image_cat),]
yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories <-  yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories[!is.na(yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories$age_human_cat),]
yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories <-  yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories[!is.na(yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories$age_text_cat),]

sum(is.na(yes_df_has_face_feature_removed_nan_with_age_remove_nan_for_categories$age_human_cat))



#Specifically for text#
#yes_df_nan_handled <- yes_df_has_face_feature
yes_df_nan_handled$age_text_cat <-cut(yes_df_nan_handled$Age, c(14,19,23,34,46,60))
yes_df_nan_handled$age_image_cat <-cut(yes_df_nan_handled$facepp_img_results_ES_faces_0_attributes_age_value, c(14,19,23,34,46,60))
yes_df_nan_handled$age_human_cat <-cut(yes_df_nan_handled$Human.judgement.for.age, c(14,19,23,34,46,60))

names(no_df_has_face_feature_removed_nan)
no_df_has_face_feature_removed_nan$age_text_cat <-cut(no_df_has_face_feature_removed_nan$Age, c(14,19,23,34,46,60))
no_df_has_face_feature_removed_nan$age_image_cat <-cut(no_df_has_face_feature_removed_nan$facepp_img_results_ES_faces_0_attributes_age_value, c(14,19,23,34,46,60))
no_df_has_face_feature_removed_nan$age_human_cat <-cut(no_df_has_face_feature_removed_nan$Human.judgement.for.age, c(14,19,23,34,46,60))



dim(yes_df_nan_handled)
summary(yes_df_nan_handled)
predictions <- yes_df_nan_handled[,"age_image_cat"]#age_image_cat
actual <- yes_df_nan_handled[,"age_human_cat"]
confusionMatrix(predictions, actual)
predictions


#--------------no-----------------------
no_df_has_face_feature <- filter(no_df, face_pp_type == 'Face_found_in_media' | face_pp_type == "Face_found_in_profile")
#no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_media")
no_df_has_face_feature_removed_nan <-  no_df_has_face_feature[!is.na(no_df_has_face_feature$Description..How.you.come.up.with.the.age.),]

#converting continous data to factos
no_df_has_face_feature_removed_nan_with_age <- filter(no_df_has_face_feature_removed_nan, no_df_has_face_feature_removed_nan$Human.judgement.for.age > 0)
no_df_has_face_feature_removed_nan_with_age$age_text_cat <-cut(no_df_has_face_feature_removed_nan_with_age$Age, c(14,19,23,34,46,60))
no_df_has_face_feature_removed_nan_with_age$age_image_cat <-cut(no_df_has_face_feature_removed_nan_with_age$facepp_img_results_ES_faces_0_attributes_age_value, c(14,19,23,34,46,60))
no_df_has_face_feature_removed_nan_with_age$age_human_cat <-cut(no_df_has_face_feature_removed_nan_with_age$Human.judgement.for.age, c(14,19,23,34,46,60))

no_df_has_face_feature_removed_nan_with_age <-  no_df_has_face_feature_removed_nan_with_age[!is.na(no_df_has_face_feature_removed_nan_with_age$age_image_cat),]
no_df_has_face_feature_removed_nan_with_age_nan_for_categories <-  no_df_has_face_feature_removed_nan_with_age[!is.na(no_df_has_face_feature_removed_nan_with_age$age_human_cat),]
no_df_has_face_feature_removed_nan_with_age_nan_for_categories <-  no_df_has_face_feature_removed_nan_with_age_nan_for_categories[!is.na(no_df_has_face_feature_removed_nan_with_age_nan_for_categories$age_text_cat),]

sum(is.na(no_df_has_face_feature_removed_nan_with_age_nan_for_categories$age_human_cat))

no_df_has_face_feature_removed_nan_with_age_nan_for_categories
#predictions <- no_df_has_face_feature_removed_nan_with_age_nan_for_categories[,"age_text_cat"]
predictions <- no_df_has_face_feature_removed_nan_with_age_nan_for_categories[,"age_image_cat"]
#predictions <- no_df_has_face_feature_removed_nan_with_age_nan_for_categories[,"age_image_cat"]
actual <- no_df_has_face_feature_removed_nan_with_age_nan_for_categories[,"age_human_cat"]
confusionMatrix(predictions, actual)
predictions

write.csv(no_df_has_face_feature_removed_nan_with_age_nan_for_categories, file = "/home/amir/work/code/Depression_image_analysis/no_df_has_face_feature_removed_nan_with_age_nan_for_categories.csv", fileEncoding = "UTF-8")

#-----------------------------Gender Gold standard----------------------------------------------------------

yes_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "yes")
no_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "no")



names(yes_df)
#yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_media"| face_pp_type == "Face_found_in_profile")
yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_media")
yes_df_has_face_feature <-  yes_df_has_face_feature[!is.na(yes_df_has_face_feature$Human.Judge.for.Gender),]
no_df_has_face_feature <-  no_df_has_face_feature[!is.na(no_df_has_face_feature$Human.Judge.for.Gender),]

names(no_df_has_face_feature)
Gender
Human.Judge.for.Gender
facepp_img_results_ES_faces_0_attributes_gender_value

yes_df$Human.Judge.for.Gender

yes_gender <- select(yes_df_has_face_feature, Human.Judge.for.Gender,facepp_img_results_ES_faces_0_attributes_gender_value,Gender)
no_gender <- select(no_df_has_face_feature, Human.Judge.for.Gender,facepp_img_results_ES_faces_0_attributes_gender_value,Gender)

summary(yes_gender)
yes_gender <- na.omit(yes_gender)
no_gender <- na.omit(no_gender)
class(yes_gender$Gender)


head(yes_df_has_face_feature_removed_nan)
yes_gender$image_gender_converted <- sapply(yes_gender$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})
no_gender$image_gender_converted <- sapply(no_gender$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})
#yes_gender$image_gender_converted 
#yes_gender$Human.Judge.for.Gender 
summary(yes_gender$Human.Judge.for.Gender)
yes_gender <- filter(yes_gender, yes_gender$Human.Judge.for.Gender != "can not decide")
no_gender <- filter(no_gender, no_gender$Human.Judge.for.Gender != "can not decide")
summary(yes_gender$Human.Judge.for.Gender)
summary(no_gender$Human.Judge.for.Gender)

head(yes_gender)
yes_gender$Gender <- as.numeric(as.character(yes_gender$Gender))
no_gender$Gender <- as.numeric(as.character(no_gender$Gender))



yes_gender$Gender
summary(yes_gender$Gender_converted)
yes_gender
head(no_gender)
no_gender$facepp_img_results_ES_faces_0_attributes_gender_value <- sapply(no_gender$facepp_img_results_ES_faces_0_attributes_gender_value, function(x) if (x == "Male") {1} else {0})


#dim(no_gender)
sum(is.na(no_gender))
no_gender_nan_handled <- na.omit(no_gender)
sum(is.na(no_gender_nan_handled))
summary(no_gender$Gender)
summary(no_gender)
yes_gender$Gender <- sapply(yes_gender$Gender, function(x) if (x > 0 ) {0} else {1})
no_gender$Gender <- sapply(no_gender$Gender, function(x) if (x > 0 ) {0} else {1})
no_gender$Gender <- sapply(no_gender$Gender, function(x) if (x > 0 ) {0} else {1})
yes_gender$Human.Judge.for.Gender <- sapply(yes_gender$Human.Judge.for.Gender, function(x) if (x == 'Female' ) {0} else {1})
no_gender$Human.Judge.for.Gender <- sapply(no_gender$Human.Judge.for.Gender, function(x) if (x == 'Female' ) {0} else {1})
summary(yes_gender$Human.Judge.for.Gender)
no_gender_nan_handled$Gender <- sapply(no_gender_nan_handled$Gender, function(x) if (x < 0 ) {1} else {0})

head(yes_gender)
head(no_gender)
summary(no_gender$Human.Judge.for.Gender)
predictions <- no_gender[,"Gender"]#image_gender_converted
actual <- no_gender[,"Human.Judge.for.Gender"]
confusionMatrix(predictions, actual)
predictions

yes_gender
#----------------------Gender LIWC--------------
yes_df <-  yes_df[!is.na(yes_df$Human.Judge.for.Gender),]
no_df <-  no_df[!is.na(no_df$Human.Judge.for.Gender),]

yes_gender <- filter(yes_df, yes_df$Human.Judge.for.Gender != "can not decide")
no_gender <- filter(no_df, no_df$Human.Judge.for.Gender != "can not decide")

yes_gender$Human.Judge.for.Gender <- droplevels(yes_gender$Human.Judge.for.Gender)
yes_gender <- select(yes_gender, Human.Judge.for.Gender,male,female,Annotation)
str(yes_gender)

no_gender$Human.Judge.for.Gender <- droplevels(no_gender$Human.Judge.for.Gender)
no_gender <- select(no_gender,Human.Judge.for.Gender,male,female,Annotation)
no_gender

merged_genders <- rbind(yes_gender, no_gender)
Annotation <- as.factor(merged_genders$Annotation)
summary(merged_genders)
str(merged_genders)

merged_genders <-rename(merged_genders, Class = Annotation )
p_female <- ggplot(merged_genders, aes(x=Human.Judge.for.Gender, y=female, fill=Class)) + geom_boxplot()
p_female <- p_female + scale_y_continuous(limits = c(0, 3)) +labs(title = "", x = "", y = "Frequency of Female References")+theme_classic()
p_female <- p_female + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 

p_male <- ggplot(merged_genders, aes(x=Human.Judge.for.Gender, y=male, fill=Class)) + geom_boxplot()
p_male <- p_male + scale_y_continuous(limits = c(0, 3)) +labs(title = "", x = "", y = "Frequency of Male References")+theme_classic()
p_male <- p_male + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=25,  face="bold") ,
  legend.title = element_text(size=25,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face = "bold")
) 


ggarrange(p_male+ font("x.text", size = 25, face = "bold") + font("y.text", size = 25, face = "bold")  , p_female + font("x.text", size = 25, face = "bold")+font("y.text", size = 25, face = "bold") , 
          labels = c("A", "B"),
          ncol = 2, nrow = 1,
          common.legend = TRUE, legend = "top"
          )


plot_grid(p_male+ font("x.text", size = 25, face = "bold")+ font("y.text", size = 25, face = "bold")  , p_female + font("x.text", size = 25, face = "bold")+ font("y.text", size = 25, face = "bold"), 
          labels = c("A", "B"),
          ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")
        
#----------------------------------Ethnicity Gold Standard-----------------------------------
yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_media"| face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_media"| face_pp_type == "Face_found_in_profile")
yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df,face_pp_type == "Face_found_in_profile")

names(yes_df_has_face_feature)
summary(no_df_has_face_feature)
summary(no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value)
summary(yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value)

yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value
round(prop.table(table(yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value,yes_df_has_face_feature$Human.Judge.for.Gender)))
#prop.table(table(no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value))

df_reordered_location_handled
head(yes_df_has_face_feature)
#facepp_img_results_ES_faces_0_attributes_gender_value
#Annotation
#age_text_cat_yes
#Human.Judge.for.Gender
#facepp_img_results_ES_faces_0_attributes_ethnicity_value
yes_df_has_face_feature <- filter(yes_df_has_face_feature, yes_df_has_face_feature$Human.Judge.for.Gender != "can not decide")
no_df_has_face_feature <- filter(no_df_has_face_feature, no_df_has_face_feature$Human.Judge.for.Gender != "can not decide")
yes_df_has_face_feature$age_human_cat <-cut(yes_df_has_face_feature$Human.judgement.for.age, c(14,19,23,34,46,60))
no_df_has_face_feature$age_human_cat <-cut(no_df_has_face_feature$Human.judgement.for.age, c(14,19,23,34,46,60))
head(yes_df_has_face_feature)


percentage_demographic = subset(yes_df_has_face_feature, select = c(age_human_cat,Human.Judge.for.Gender,facepp_img_results_ES_faces_0_attributes_ethnicity_value))

no_gender <- filter(no_gender, no_gender$Human.Judge.for.Gender != "can not decide")
yes_df_has_face_feature$Human.Judge.for.Gender

#------------Function for showing every column as percentage and count--------------
tblFun <- function(x){
  tbl <- table(x)
  res <- cbind(tbl,round(prop.table(tbl)*100,2))
  colnames(res) <- c('Count','Percentage')
  res
}
do.call(rbind,lapply(percentage_demographic,tblFun))
#-------------------------------------------------------------------------
#------------------drawing 2 way_contigency table-------------
#1st:  Gender, EThinicty
mytable_yes <- table( yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value, droplevels(yes_df_has_face_feature$Human.Judge.for.Gender)) 
ftable(mytable_yes)
prop.table(mytable_yes)*100

mytable_no <- table( no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value,droplevels(no_df_has_face_feature$Human.Judge.for.Gender) ) 
ftable(mytable_no)
prop.table(mytable_no)*100

#colours <- c("red", "orange", "blue", "yellow", "green")
library(RColorBrewer)
display.brewer.all()
barplot(prop.table(mytable_no)*100 ,beside = T, main = "Gender vs Ethinicty",col=brewer.pal(3,"Blues"))
legend("topright", c("Asian","Black","White"), cex=0.8, bty="n", fill =brewer.pal(3,"Blues"))

#--------------------------------Merging for 3 way

deppressed <- data.frame(Race= yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value,
                         Gender =yes_df_has_face_feature$Human.Judge.for.Gender, 
                         Annotation=yes_df_has_face_feature$Annotation)
controled <- data.frame(Race= no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value,
                        Gender =no_df_has_face_feature$Human.Judge.for.Gender, 
                        Annotation=no_df_has_face_feature$Annotation)
head(deppressed)
merged_age_categories <- rbind(deppressed, controled)
Annotation <- as.factor(merged_age_categories$Annotation)

#merged_age_categories$Annotation <- sapply(merged_age_categories$Annotation, function(x) if (x == 'no' ) {"Control"} else {"Depressed"})
#merged_age_categories <-rename(merged_age_categories, Class = Annotation )
#balloonplot(t(mytable), main ="housetasks", xlab ="", ylab="", label = FALSE, show.margins = FALSE)
summary(merged_age_categories)

mytable <- table(merged_age_categories$Annotation, merged_age_categories$Race, merged_age_categories$Gender)

#--------------Avg. based on ethnicity --------------------
mytable <- xtabs(~Annotation+Race+Gender, merged_age_categories)
summary(mytable)
library("gplots")
mytable <- table(merged_age_categories$Annotation, merged_age_categories$Race)

mytable_prop <- prop.table ( mytable,2)*100
mytable_prop
#-------------------Race-------------------
mytable <- xtabs(~Annotation+Race+Gender, merged_age_categories)
summary(mytable)
mytable_prop <- prop.table ( mytable,1)*100
mytable_prop
#---------------------Gender---------------
mytable <- xtabs(~Annotation+Race+Gender, merged_age_categories)
summary(mytable)
mytable_prop <- prop.table ( mytable,3)*100
mytable_prop
#-------------------------------------------------------
mytable <- table(merged_age_categories$Annotation, merged_age_categories$Race)
mytable
mytable_prop <- prop.table ( mytable,2)*100
chisq.test(mytable_prop)
ftable(mytable_prop)
#chisq.test(mytable)
#----------------------------------------
# ----------Categorical data and chi_squared
#age
yes_df_has_face_feature$age_human_cat
yes_df_nan_handled <-  yes_df[!is.na(yes_df$Human.judgement.for.age),]
yes_df_nan_handled <- filter(yes_df_nan_handled, yes_df_nan_handled$Human.judgement.for.age > 0)
yes_df_nan_handled$age_human_cat <-cut(yes_df_nan_handled$Human.judgement.for.age, c(14,19,23,34,46,60))
summary(yes_df_nan_handled$age_human_cat)
yes_df_nan_handled <-  yes_df_nan_handled[!is.na(yes_df_nan_handled$age_human_cat),]


no_df_nan_handled <-  no_df[!is.na(no_df$Human.judgement.for.age),]
no_df_nan_handled <- filter(no_df_nan_handled, no_df_nan_handled$Human.judgement.for.age > 0)
no_df_nan_handled$age_human_cat <-cut(no_df_nan_handled$Human.judgement.for.age, c(14,19,23,34,46,60))
summary(no_df_nan_handled$age_human_cat)
no_df_nan_handled <-  no_df_nan_handled[!is.na(no_df_nan_handled$age_human_cat),]


deppressed <- data.frame(
                         Age =yes_df_nan_handled$age_human_cat, 
                         Annotation=yes_df_nan_handled$Annotation)
controled <- data.frame(
                        Age =no_df_nan_handled$age_human_cat, 
                        Annotation=no_df_nan_handled$Annotation)

head(deppressed)
merged_age_categories <- rbind(deppressed, controled)
#summary(merged_age_categories$Age)
Annotation <- as.factor(merged_age_categories$Annotation)

chisq
tem_table <- table(merged_age_categories$Annotation, merged_age_categories$Age)
tem_table
chisq <- chisq.test(tem_table)
chisq$expected
corrplot(chisq$residuals, is.cor = FALSE)

# Contibution in percentage (%)
contrib <- 100*chisq$residuals^2/chisq$statistic
round(contrib, 3)
contrib
corrplot(contrib, is.cor = FALSE)

#-------------------Chi squared for gender----------------

no_df <-  no_df[!is.na(no_df$Human.Judge.for.Gender),]
head(no_df$Human.Judge.for.Gender)
no_df <- filter(no_df, no_df$Human.Judge.for.Gender != "can not decide")

yes_df <-  yes_df[!is.na(yes_df$Human.Judge.for.Gender),]
head(yes_df$Human.Judge.for.Gender)
yes_df <- filter(yes_df, yes_df$Human.Judge.for.Gender != "can not decide")
deppressed <- data.frame(
  Gender =yes_df$Human.Judge.for.Gender, 
  Annotation=yes_df$Annotation)
controled <- data.frame(
  Gender =no_df$Human.Judge.for.Gender, 
  Annotation=no_df$Annotation)

merged_gender<- rbind(deppressed, controled)
Annotation <- as.factor(merged_gender$Annotation)
tem_table <- table(merged_gender$Annotation, droplevels(merged_gender$Gender))
tem_table
chisq <- chisq.test(tem_table)
chisq
corrplot(chisq$residuals, is.cor = FALSE)
dim(merged_gender)

#-------------------Chi squared for Ethicity----------------
no_df <-  no_df[!is.na(no_df$facepp_img_results_ES_faces_0_attributes_ethnicity_value),]
head(no_df$facepp_img_results_ES_faces_0_attributes_ethnicity_value)
#no_df <- filter(no_df, no_df$Human.Judge.for.Gender != "can not decide")

yes_df <-  yes_df[!is.na(yes_df$facepp_img_results_ES_faces_0_attributes_ethnicity_value),]
head(yes_df$facepp_img_results_ES_faces_0_attributes_ethnicity_value)
#yes_df <- filter(yes_df, yes_df$Human.Judge.for.Gender != "can not decide")

deppressed <- data.frame(
  Race =yes_df$facepp_img_results_ES_faces_0_attributes_ethnicity_value, 
  Annotation=yes_df$Annotation)
controled <- data.frame(
  Race =no_df$facepp_img_results_ES_faces_0_attributes_ethnicity_value, 
  Annotation=no_df$Annotation)

merged_ethnicity <- rbind(deppressed, controled)
Annotation <- as.factor(merged_ethnicity$Annotation)
tem_table <- table(merged_ethnicity$Annotation, droplevels(merged_ethnicity$Race))
tem_table
chisq <- chisq.test(tem_table)
chisq
corrplot(chisq$residuals, is.cor = FALSE)
dim(merged_ethnicity)
#----------------------------Image features Exploration-----------------------------
yes_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "Depressed")
no_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "Control")
names(yes_df)

yes_df_profile_image_features <- select(yes_df, Annotation,starts_with("profile_"))
not_df_profile_image_features <- select(no_df, Annotation ,starts_with("profile_"))
merged_image_numeric <- rbind(yes_df_profile_image_features, not_df_profile_image_features)
sum(is.na(merged_image_numeric))

yes_df_tweet_imageFeatures <- select(yes_df, Annotation,starts_with("tweet_imageFeatures"))
not_df_tweet_imageFeatures <- select(no_df, Annotation ,starts_with("tweet_imageFeatures"))
merged_tweet_imageFeatures <- rbind(yes_df_tweet_imageFeatures, not_df_tweet_imageFeatures)
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, Class = Annotation )
sum(is.na(merged_tweet_imageFeatures))

yes_df_has_face_feature <- filter(yes_df, face_pp_type == "Face_found_in_profile")
no_df_has_face_feature <- filter(no_df, face_pp_type == "Face_found_in_profile")
summary(yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_blur_blurness_value)
summary(no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_blur_blurness_value)

merged_df <- rbind(yes_df, no_df)
names(merged_df)
merged_df <-rename(merged_df, Class = Annotation )
merged_df_plot <- ggplot(merged_df, aes(y=profile_imageFeatures.averageRGB, x=Annotation)) + geom_boxplot()
merged_df_plot + stat_summary(fun.y=mean, geom="point", shape=23, size=4)

summary(yes_df)
summary(no_df)

#imputation by mean
merged_image_numeric[] <- lapply(merged_image_numeric, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
sum(is.na(merged_image_numeric))
merged_image_numeric <-rename(merged_image_numeric, Class = Annotation )

head(yes_df)


yes_df_sample <- sample_n(yes_df, 250)
no_df_sample <- sample_n(no_df, 250)
t_test <- t.test(yes_df_sample$favourites_count ,no_df_sample$favourites_count )
p <- t_test$p.value
p <- p.adjust(p, method = "bonferroni")
t_test
p

yes_df_sample<- yes_df_sample[1:183]
merged_df_sample <- rbind(no_df_sample,yes_df_sample)


pp <- ggplot(merged_df_sample, aes(x=description_sentiment_polarity, colour=Annotation)) + geom_density()+ labs(x = "description_sentiment_polarity", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))+ 
  theme_classic()#+ theme(axis.text=element_text(size=12),
pp


if (p < 0.001){
    print ('***')
  }else if (p < 0.01){
    print ('**')
  } else if (p < 0.05){
     print ('*')
 }

#----------------------------------------------
p <- ggplot(merged_df, aes(x=profile_imageFeatures.colorfulness, colour=Annotation)) + geom_density()+ labs(x = "Clout", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))+ 
  theme_classic()#+ theme(axis.text=element_text(size=12),
p + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15)
)                        #axis.title=element_text(size=14,face="bold"))


#---------------------------------------------------

#, appealing images
#tend to have increased contrast, sharpness, saturation and less
#blur, which is the case for people high in openness.

#scale_y_continuous(limits = c(0, 3)) +labs(title = "", x = "", y = "Frequency of Female References")+
summary(merged_tweet_imageFeatures$tweet_imageFeatures.sharpness)

merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_sharpness = tweet_imageFeatures.sharpness)
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_naturalness = tweet_imageFeatures.naturalness  )
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_saturation_mean = tweet_imageFeatures.saturationChannelMean  )
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_averageRGB = tweet_imageFeatures.averageRGB )
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_colorfulness  = tweet_imageFeatures.colorfulness )
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_brightness = tweet_imageFeatures.brightness  )
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_GrayScaleMean = tweet_imageFeatures.GrayScaleMean  )
merged_tweet_imageFeatures <-rename(merged_tweet_imageFeatures, shared_image_hueChannelMean = tweet_imageFeatures.hueChannelMean   )


tweet_imageFeatures.sharpness <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_sharpness, x=Class)) + geom_boxplot()
tweet_imageFeatures.sharpness <- tweet_imageFeatures.sharpness + theme_classic()
tweet_imageFeatures.sharpness <- tweet_imageFeatures.sharpness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.sharpness <- tweet_imageFeatures.sharpness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.sharpness

tweet_imageFeatures.naturalness <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_naturalness, x=Class)) + geom_boxplot()
tweet_imageFeatures.naturalness <- tweet_imageFeatures.naturalness + theme_classic()
tweet_imageFeatures.naturalness <- tweet_imageFeatures.naturalness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.naturalness <- tweet_imageFeatures.naturalness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.naturalness

tweet_imageFeatures.saturationChannelMean <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_saturation_mean, x=Class)) + geom_boxplot()
tweet_imageFeatures.saturationChannelMean <- tweet_imageFeatures.saturationChannelMean + theme_classic()
tweet_imageFeatures.saturationChannelMean <- tweet_imageFeatures.saturationChannelMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.saturationChannelMean <- tweet_imageFeatures.saturationChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.saturationChannelMean



tweet_imageFeatures.averageRGB <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_averageRGB, x=Class)) + geom_boxplot()
tweet_imageFeatures.averageRGB <- tweet_imageFeatures.averageRGB + theme_classic()
tweet_imageFeatures.averageRGB <- tweet_imageFeatures.averageRGB + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.averageRGB <- tweet_imageFeatures.averageRGB  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.averageRGB


tweet_imageFeatures.colorfulness <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_colorfulness, x=Class)) + geom_boxplot()
tweet_imageFeatures.colorfulness <- tweet_imageFeatures.colorfulness + theme_classic()
tweet_imageFeatures.colorfulness <- tweet_imageFeatures.colorfulness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.colorfulness <- tweet_imageFeatures.colorfulness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.colorfulness


tweet_imageFeatures.brightness <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_brightness, x=Class)) + geom_boxplot()
tweet_imageFeatures.brightness <- tweet_imageFeatures.brightness + theme_classic()
tweet_imageFeatures.brightness <- tweet_imageFeatures.brightness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.brightness <- tweet_imageFeatures.brightness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.brightness

tweet_imageFeatures.GrayScaleMean <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_GrayScaleMean, x=Class)) + geom_boxplot()
tweet_imageFeatures.GrayScaleMean <- tweet_imageFeatures.GrayScaleMean + theme_classic()
tweet_imageFeatures.GrayScaleMean <- tweet_imageFeatures.GrayScaleMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.GrayScaleMean <- tweet_imageFeatures.GrayScaleMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.GrayScaleMean

tweet_imageFeatures.hueChannelMean <- ggplot(merged_tweet_imageFeatures, aes(y=shared_image_hueChannelMean, x=Class)) + geom_boxplot()
tweet_imageFeatures.hueChannelMean <- tweet_imageFeatures.hueChannelMean + theme_classic()
tweet_imageFeatures.hueChannelMean <- tweet_imageFeatures.hueChannelMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.hueChannelMean <- tweet_imageFeatures.hueChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.hueChannelMean

tweet_imageFeatures.saturationChannelVAR <- ggplot(merged_tweet_imageFeatures, aes(y=tweet_imageFeatures.saturationChannelVAR, x=Class)) + geom_boxplot()
tweet_imageFeatures.saturationChannelVAR <- tweet_imageFeatures.saturationChannelVAR + theme_classic()
tweet_imageFeatures.saturationChannelVAR <- tweet_imageFeatures.saturationChannelVAR + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=15, face="bold"),
  axis.title.y = element_text(color="black", size=15, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 10),
  axis.text.y = element_text(size = 10, face= "bold")
) 
tweet_imageFeatures.saturationChannelVAR <- tweet_imageFeatures.saturationChannelVAR  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
tweet_imageFeatures.saturationChannelVAR


#facepp_img_results_ES_faces_0_attributes_blur_blurness_value

names(merged_tweet_imageFeatures)
ggarrange(tweet_imageFeatures.saturationChannelVAR,tweet_imageFeatures.hueChannelMean,tweet_imageFeatures.GrayScaleMean,tweet_imageFeatures.brightness,tweet_imageFeatures.colorfulness,tweet_imageFeatures.saturationChannelMean ,tweet_imageFeatures.sharpness, tweet_imageFeatures.naturalness, 
          #labels = c("A", "B"),
          ncol = 4, nrow = 2,
          common.legend = TRUE, legend = "top"
)





#In terms of colors, conscientious users prefer pictures
#which are not grayscale and are overall more colorful, natural
#and bright.
GrayScale <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.GrayScaleMean, x=Class)) + geom_boxplot()
GrayScale <- GrayScale + theme_classic()
GrayScale <- GrayScale + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
GrayScale <- GrayScale  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
GrayScale

averageRGB <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.averageRGB, x=Class)) + geom_boxplot()
averageRGB <- averageRGB + theme_classic()
averageRGB <- averageRGB + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
averageRGB <- averageRGB  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
averageRGB


colorfulness <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.colorfulness, x=Class)) + geom_boxplot()
colorfulness <- colorfulness + theme_classic()
colorfulness <- colorfulness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
colorfulness <- colorfulness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
colorfulness


saturationChannelVAR <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.saturationChannelVAR, x=Class)) + geom_boxplot()
saturationChannelVAR <- saturationChannelVAR + theme_classic()
saturationChannelVAR <- saturationChannelVAR + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
saturationChannelVAR <- saturationChannelVAR  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
saturationChannelVAR



saturationChannelMean <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.saturationChannelMean, x=Class)) + geom_boxplot()
saturationChannelMean <- saturationChannelMean + theme_classic()
saturationChannelMean <- saturationChannelMean + theme(
  plot.title = element_text(color="black", size=20, face="bold.italic"),
  axis.title.x = element_text(color="black", size=20, face="bold"),
  axis.title.y = element_text(color="black", size=20, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
saturationChannelMean <- saturationChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
saturationChannelMean





brightness <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.brightness, x=Class)) + geom_boxplot()
brightness <- brightness + theme_classic()
brightness <- brightness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
brightness <- brightness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
brightness


contrast <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.contrast, x=Class)) + geom_boxplot()
contrast <- contrast + theme_classic()
contrast <- contrast + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
contrast <- contrast  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
contrast


sharpness <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.sharpness, x=Class)) + geom_boxplot()
sharpness <- sharpness + theme_classic()
sharpness <- sharpness + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
sharpness <- sharpness  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
sharpness


natural <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.naturalness, x=Class)) + geom_boxplot()
natural <- natural + theme_classic()
natural <- natural + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
natural <- natural  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
natural

hueChannelMean <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.hueChannelMean, x=Class)) + geom_boxplot()
hueChannelMean <- hueChannelMean + theme_classic()
hueChannelMean <- hueChannelMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
hueChannelMean <- hueChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
hueChannelMean

RedChannelMean <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.RedChannelMean, x=Class)) + geom_boxplot()
RedChannelMean <- RedChannelMean + theme_classic()
RedChannelMean <- RedChannelMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
RedChannelMean <- RedChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
RedChannelMean

BlueChannelMean <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.BlueChannelMean, x=Class)) + geom_boxplot()
BlueChannelMean <- BlueChannelMean + theme_classic()
BlueChannelMean
BlueChannelMean <- BlueChannelMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
BlueChannelMean <- BlueChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
BlueChannelMean


GreenChannelMean <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.GreenChannelMean, x=Class)) + geom_boxplot()
GreenChannelMean <- GreenChannelMean + theme_classic()
GreenChannelMean <- GreenChannelMean + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
GreenChannelMean <- GreenChannelMean  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
GreenChannelMean
 


ggarrange(RedChannelMean,BlueChannelMean,GreenChannelMean,sharpness,saturationChannelMean ,natural, hueChannelMean, contrast , saturationChannelVAR, brightness, GrayScale, 
          #labels = c("A", "B"),
          ncol = 4, nrow = 3,
          common.legend = TRUE, legend = "top"
)


ggarrange(averageRGB,sharpness ,natural, hueChannelMean, contrast , saturationChannelMean, brightness, GrayScale, 
          #labels = c("A", "B"),
          ncol = 4, nrow = 3,
          common.legend = TRUE, legend = "top"
)

#contrast, sharpness, brightness, 
ggarrange(averageRGB ,natural, colorfulness,  saturationChannelMean,
          #labels = c("A", "B"),
          ncol = 4, nrow = 1,
          common.legend = TRUE, legend = "top"
)


t_test <- t.test(yes_df$profile_imageFeatures.hueChannelVAR, no_df$profile_imageFeatures.hueChannelVAR)
t_test
p <- t_test$p.value
p
p <- p.adjust(p, method = "bonferroni")
p

if (p < 0.001){
  print ('***')
}else if (p < 0.01){
  print ('**')
} else if (p < 0.05){
  print ('*')
}


#plot_grid(p_male+ font("x.text", size = 25, face = "bold")+ font("y.text", size = 25, face = "bold")  , p_female + font("x.text", size = 25, face = "bold")+ font("y.text", size = 25, face = "bold"), 
   #       labels = c("A", "B"),
   #       ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom")


yes_df$profile_imageFeatures.BlueChannelMean
str(yes_df)
summarize(df_reordered_loc_anotated, age_text = mean(Age, na.rm = TRUE), gender_text =mean(Gender, na.rm = TRUE) , age_image= mean(facepp_img_results_ES_faces_0_attributes_age_value, na.rm = TRUE), face_pp_type)
summarize(df_reordered_loc_anotated, face_pp_type)

#Getting Chi-squared value for image extraction
#remove Nans
df_face_pp_not_nan <-  df_reordered_location_handled[!is.na(df_reordered_location_handled$face_pp_type),]
df_face_pp_not_nan_subset <- subset(df_face_pp_not_nan, face_pp_type == 'Face_NOT_found_in_profile_or_in_media')
df_face_pp_not_vs_annotation <- subset(df_face_pp_not_nan_subset, select = c(face_pp_type,Annotation))
table(df_face_pp_not_vs_annotation)
df_face_pp_not_vs_annotation <- table(df_anotaion_selecterd_2$face_pp_type, df_anotaion_selecterd_2$Annotation )
#remove levels
df_face_pp_not_vs_annotation <- table(droplevels(df_anotaion_selecterd_2$face_pp_type),df_anotaion_selecterd_2$Annotation)
df_face_pp_not_vs_annotation

if (chisq.test(df_face_pp_not_vs_annotation)$p.value < 0.001){
  print ('***')
}else if (chisq.test(df_face_pp_not_vs_annotation)$p.value < 0.01){
  print ('**')
} else if (chisq.test(df_face_pp_not_vs_annotation)$p.value < 0.05){
  print ('*')
}
chisq.test(df_face_pp_not_vs_annotation)

#---------------------Anova anlaysis for age----------------

yes_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "Depressed")
no_df<- filter(df_reordered_location_handled, df_reordered_location_handled$Annotation == "Control")

yes_df_nan_handled$human_age_cat <-cut(yes_df_nan_handled$Human.judgement.for.age, c(11,19,23,34,46,60))
yes_df_nan_handled$human_age_cat

no_df_has_face_feature_removed_nan$age_text_cat_no <-cut(no_df_has_face_feature_removed_nan$Age, c(14,19,23,34,46,60))
summary(yes_df_has_face_feature_removed_nan_with_age$age_text_cat)

names(yes_df_nan_handled)

group_by(yes_df_nan_handled, human_age_cat) %>%
  summarise(
    count = n(),
    mean = mean(swear, na.rm = TRUE),
    sd = sd(swear, na.rm = TRUE)
  )

merged_df <- rbind(yes_df, no_df)
merged_df$human_age_cat <-cut(merged_df$Human.judgement.for.age, c(11,19,23,34,46,60))
merged_df <-  merged_df[!is.na(merged_df$human_age_cat),]

group_by(merged_df, human_age_cat) %>%
  summarise(
    count = n(),
    mean = mean(profile_imageFeatures.colorfulness, na.rm = TRUE),
    sd = sd(profile_imageFeatures.colorfulness, na.rm = TRUE)
  )

yes_df$human_age_cat <-cut(yes_df$Human.judgement.for.age, c(11,19,23,34,46,60))
yes_df <-  yes_df[!is.na(yes_df$human_age_cat),]

yes_df_age <- ggplot(yes_df, aes(x=human_age_cat, y=Authentic)) + geom_boxplot()
yes_df_age




head(merged_df$Annotation)
ggboxplot(yes_df_nan_handled, x = human_age_cat, y = Analytic)
Analytic_age2 <- ggplot(yes_df_nan_handled, aes(x=human_age_cat, y=Analytic)) + geom_boxplot()
Analytic_age2
#, palette = c("#00AFBB", "#E7B800", "#FC4E07"),
#order = c("(11,19]", "(19,23]", "(23,34]","(34,46]","(46,60]"),
#ylab = "friend", xlab = "human_age_cat")





#------------Combined__________
merged_df <-rename(merged_df, Class = Annotation )
merged_df <-rename(merged_df, Age_Groups = human_age_cat )

Analytic_age <- ggplot(merged_df, aes(x=Age_Groups, y=Analytic, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
#+ geom_jitter(position = position_jitter(0.1))
Analytic_age
Analytic_age <- Analytic_age + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=25,  face="bold") ,
  legend.title = element_text(size=25,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face = "bold")
) 
Analytic_age <- Analytic_age  
Analytic_age

Authentic_age <- ggplot(merged_df, aes(x=Age_Groups, y=Authentic, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
Authentic_age


Clout_age <- ggplot(merged_df, aes(x=Age_Groups, y=Clout, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
Clout_age


article_age <- ggplot(merged_df, aes(x=Age_Groups, y=article, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
article_age

Sixltr_age <- ggplot(merged_df, aes(x=Age_Groups, y=Sixltr, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
Sixltr_age

cogproc_age <- ggplot(merged_df, aes(x=Age_Groups, y=cogproc, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
cogproc_age

swear_age <- ggplot(merged_df, aes(x=Age_Groups, y=swear, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
swear_age <- swear_age + ylim(0, 3)
swear_age

  
ggarrange(Analytic_age, Authentic_age,Clout_age,Sixltr_age,swear_age, cogproc_age,
          #labels = c("A", "B"),
          ncol = 3, nrow = 2,
          common.legend = TRUE, legend = "top"
)

merged_df2 <-rename(merged_df, self_references = i )
self_reference_age <- ggplot(merged_df2, aes(x=Age_Groups, y=self_references , fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
self_reference_age


work_age <- ggplot(merged_df, aes(x=Age_Groups, y=work, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
work_age

money_age <- ggplot(merged_df, aes(x=Age_Groups, y=money, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
money_age <- money_age  + ylim(0, 2)

Dic_age <- ggplot(merged_df, aes(x=Age_Groups, y=Dic, fill=Class))  + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .65)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
Dic_age

netspeak_age <- ggplot(merged_df, aes(x=Age_Groups, y=netspeak, fill=Class))  + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .65)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
netspeak_age <- netspeak_age+ ylim(0, 10)

tweet_imageFeatures.saturationChannelMean_age <- ggplot(merged_df, aes(x=Age_Groups, y=tweet_imageFeatures.saturationChannelMean, fill=Class))  + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .65)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
tweet_imageFeatures.saturationChannelMean_age

tweet_imageFeatures.naturalness_age <- ggplot(merged_df, aes(x=Age_Groups, y=tweet_imageFeatures.naturalness, fill=Class))  + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .65)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
tweet_imageFeatures.naturalness_age

tweet_imageFeatures.hueChannelVAR_age <- ggplot(merged_df, aes(x=Age_Groups, y=tweet_imageFeatures.hueChannelVAR, fill=Class))  + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .65)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
tweet_imageFeatures.hueChannelVAR_age



tweet_imageFeatures.colorfulness_age <- ggplot(merged_df, aes(x=Age_Groups, y=tweet_imageFeatures.colorfulness, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
tweet_imageFeatures.colorfulness_age

Dic_age <- ggplot(merged_df, aes(x=Age_Groups, y=Dic, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
Dic_age

body_age <- ggplot(merged_df, aes(x=Age_Groups, y=body, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
body_age <- body_age+ ylim(0,3.5)

sexual_age <- ggplot(merged_df, aes(x=Age_Groups, y=sexual, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
sexual_age <-sexual_age + ylim(0,1.5)


profile_imageFeatures.naturalness_age <- ggplot(merged_df, aes(x=Age_Groups, y=profile_imageFeatures.naturalness, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
profile_imageFeatures.naturalness_age 

tweet_imageFeatures.naturalness_age <- ggplot(merged_df, aes(x=Age_Groups, y=tweet_imageFeatures.naturalness, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + theme(legend.position="none")
tweet_imageFeatures.naturalness_age 

ggarrange(Dic_age , self_reference_age,work_age,profile_imageFeatures.naturalness_age ,tweet_imageFeatures.naturalness_age,tweet_imageFeatures.colorfulness_age,
          #labels = c("A", "B"),
          ncol = 3, nrow = 2,
          common.legend = TRUE, legend = "top"
)


ggarrange(body_age , sexual_age, tweet_imageFeatures.saturationChannelMean_age,
          #labels = c("A", "B"),
          ncol = 3, nrow = 1,
          common.legend = TRUE, legend = "top"
)

# Compute the analysis of variance
res.aov <- aov(profile_imageFeatures.GrayScaleMean ~ human_age_cat, data = merged_df)
# Summary of the analysis
summary(res.aov)


# Compute the analysis of variance
res.aov <- aov(ingest ~ human_age_cat, data = yes_df)
# Summary of the analysis
summary(res.aov)

#---------------Contigency Gender and Age----------------------
merged_df_filtered <- filter(merged_df, merged_df$Human.Judge.for.Gender != "can not decide")
mytable <- xtabs(~Annotation+human_age_cat+droplevels(Human.Judge.for.Gender), merged_df_filtered)
summary(mytable)
mytable_prop <- prop.table ( mytable,3)*100
mytable_prop


mytable <- xtabs(~Annotation+human_age_cat+droplevels(Human.Judge.for.Gender), merged_df_filtered)
summary(mytable)
mytable_prop <- prop.table ( mytable,2)*100
mytable_prop

mytable <- xtabs(~human_age_cat+face_pp_type, yes_df)
summary(mytable)
mytable_prop <- prop.table ( mytable,1)*100
mytable_prop

mytable <- xtabs(~Annotation+ human_age_cat+face_pp_type, merged_df)
summary(mytable)
mytable_prop <- prop.table ( mytable,2)*100
mytable_prop

#---------------------------------------------------
#no_df_has_face_feature$X which facepp_img_results_ES_faces_0_attributes_ethnicity_value == "Asian"
asian_no <- no_df_has_face_feature[which(no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value == "Asian"),]
asian_no$X

asian_no <- no_df_has_face_feature[which(no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value == "Asian"),]
asian_no$X


asian_yes <- yes_df_has_face_feature[which(yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value == "Black"),]

asian_yes$X

deppressed <- data.frame(Enthnicity= yes_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value,
                         Annotation=yes_df_has_face_feature$Annotation)
controled <- data.frame(Enthnicity= no_df_has_face_feature$facepp_img_results_ES_faces_0_attributes_ethnicity_value,
                        Annotation=no_df_has_face_feature$Annotation)

summary(deppressed)
merged_ethnicity_categories <- rbind(deppressed, controled)
Annotation <- as.factor(merged_ethnicity_categories$Annotation)
Enthnicity <- as.factor(merged_ethnicity_categories$Annotation)

merged_ethnicity_categories$Annotation <- sapply(merged_ethnicity_categories$Annotation, function(x) if (x == 'no' ) {"Control"} else {"Depressed"})
merged_ethnicity_categories <-rename(merged_ethnicity_categories, Class = Annotation )
#PDF for depressed vs Non-depressed
summary(merged_ethnicity_categories)
ggplot(merged_ethnicity_categories, aes(x=Enthnicity, colour=Class)) + geom_density()+ labs(x = "Age Distribution", y = "PDF")  + guides(fill=guide_legend("my awesome title"))+ 
  theme_classic()+ geom_bar(aes(fill = Class))

merged_ethnicity_categories
mytable <- table(A, B, C) 
ftable(mytable)
#--------hint----------
#getting type of each column of dataframe
#sapply(d, class)


dmy <- dummyVars(" ~ .", data = df_reordered_location_handled,fullRank = T)
df_dummified <- data.frame(predict(dmy, newdata = df_reordered_location_handled))
summary(df_dummified)
str(df_dummified)

#check factors in dataframe
sort(sapply(df_reordered_location_handled,function(x) class (x) == "factor"))
class(df_dummified$profile_text_color_blue)
df_dummified$profile_text_color_blue


#--------select if numeric------
df_reordered_omitted_nans_filtered_50s_num = select_if(df_reordered_omitted_nans_filtered_50s, is.numeric)
class(df_reordered_omitted_nans_filtered_50s_num$quant)




#df_reordered_omitted_nans_filtered_50s_num <- lapply(df_reordered_omitted_nans_filtered_50s_num, as.numeric)

df_reordered_omitted_nans_filtered_50s_numeric <- df_reordered_omitted_nans_filtered_50s [,sapply(df_reordered_omitted_nans_filtered_50s, function (x) class (x)== 'numeric')]
df_reordered_omitted_nans_filtered_50s_factor <- df_reordered_omitted_nans_filtered_50s [,sapply(df_reordered_omitted_nans_filtered_50s, function (x) class (x)== 'factor')]

df_reordered_omitted_nans_filtered_50s_numeric_ <- data.frame(sapply(df_reordered_omitted_nans_filtered_50s_numeric, function(x) as.numeric(as.character(x))))

str(df_reordered_omitted_nans_filtered_50s_num)


#-----------------------------------------------------Imputation----------------
sapply(df_dummified, mode)
sapply(df_dummified, class)


df_dummified$Annotation.yes
nbdim <- estim_ncpPCA(df_dummified) # estimate the number of dimensions to impute
df_dummified_imputed <- MIPCA(df_dummified, ncp = nbdim, nboot = 1000)

str(df_reordered_omitted_nans_filtered_50s_num)




nb <- estim_ncpMCA(df_reordered_omitted_nans_filtered_50s,ncp.max=5) ## Time-consuming, nb = 4
df_reordered_omitted_nans_filtered_50s.res <- MIMCA(df_reordered_omitted_nans_filtered_50s, ncp=4,nboot=10)


df_reordered_omitted_nans_filtered_50s %>% drop_na(profile_link_color, profile_background_color, profile_sidebar_border_color, profile_text_color, profile_sidebar_fill_color,
                                                   pigeo_results_country, pigeo_results_city,pigeo_results_state,face_pp_type, 
                                                   facepp_img_results_ES_faces_0_attributes_ethnicity_value, facepp_img_results_ES_faces_0_attributes_gender_value,
                                                   facepp_img_results_ES_faces_0_attributes_glass_value,Annotation)

str(df_reordered_omitted_nans_filtered_50s)

#sum(is.na(df_reordered_omitted_nans$Annotation))
#facepp_img_results_ES_faces_0_attributes_blur_gaussianblur_threshold
#facepp_img_results_ES_faces_0_attributes_blur_motionblur_threshold,

df_reordered_omitted_nans_filtered_50s_pmm <- mice(df_reordered_omitted_nans_filtered_50s, m=5, maxit = 50, method = 'pmm', seed = 500)
iris.mis$imputed_age <- with(df_reordered_omitted_nans_filtered_50s, impute(df_reordered_omitted_nans_filtered_50s, mean))

df_reordered_omitted_nans_filtered_50s_mediann <- with(df_reordered_omitted_nans_filtered_50s, impute(df_reordered_omitted_nans_filtered_50s, median))
df_reordered_omitted_nans_filtered_50s_knn_imputed = preProcess(df_reordered_omitted_nans_filtered_50s, "medianImpute")

summary(df_reordered_omitted_nans_filtered_50s_knn_imputed)

df_reordered_omitted_nans_filtered_50s_median = preProcess(df_reordered_omitted_nans_filtered_50s, "medianImpute")
#df_reordered_omitted_nans_filtered_50s.miss.model = predict(df_reordered_omitted_nans, df_reordered_omitted_nans_filtered_50s)


str(df.miss.pred)


str(df_reordered) 
sum(is.na(df_reordered_omitted_nans))
#visualization
transparentTheme(trans = .4)

featurePlot(x = df[c(14,19)], 
            y = df$dim_market, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

#imputation by Hmsic
str(df)
library(Hmisc)

df.mis.model = preProcess(df, "knnImpute")
df.miss.pred = predict(df.mis.model, df)
str(df.miss.pred)

sum(is.na(df.miss.pred))

#check the result of imputation
sort(sapply(df,function(x) sum(is.na(x))))
missmap(df, main = "Missing values vs observed")


#----------------Preprocessing------------------------
#warning: we can not have both standarization(scaling & centering) and normizination simentanously
#scaling
#scaling: calucalte the standard deviation for an attribute and divides each value by that standard deviation.
preprocessParams <- preProcess(df[7:38], method=c("scale"))
print(preprocessParams)
df_scaled <- predict(preprocessParams, df[7:38])

#centering
#centering: calculates the mean for an attribute and subtracts it from each value
preprocessParams <- preProcess(df[7:38], method=c("center"))
print(preprocessParams)
df_centered <- predict(preprocessParams, df[7:38])

#Standardize
#combining both the scale and center transforms
preprocessParams <- preProcess(df[7:38], method=c("scale","center"))
print(preprocessParams)
df_standarized <- predict(preprocessParams, df[7:38])


#normailization
#normalziing: scaled into the range of [0, 1] 
preprocessParams <- preProcess(df[7:38], method=c("range"))
print(preprocessParams)
df_normalzied <- predict(preprocessParams, df[7:38])


#pca: Transform the data to the principal components. 
#keep uncorrelated features ,good for linear and generalized linear regression
preprocessParams <- preProcess(df[7:38], method=c("pca"))
print(preprocessParams)
df_pca <- predict(preprocessParams, df[7:38])


#More preprocessing with KNN for imputation plus centering and scaling
preProcValues <- preProcess(df, method = c("knnImpute","center","scale"))
df_processed_KNN_centered_scaled <- predict(preProcValues, df)

sum(is.na(df_processed_KNN_centered_scaled))

#----------------handling categorical variables ------------------------------------
#creating dummy varibales using one hot coding, first for the dependent variable
df_processed_KNN_centered_scaled$dim_is_requested<-ifelse(df_processed_KNN_centered_scaled$dim_is_requested=='true',1,0)
#str(df_processed_KNN_centered_scaled)

#dummyVars function breaks out unique values from a column into individual columns
#df_reordered_omitted_nans
dmy <- dummyVars(" ~ .", data = df_reordered_omitted_nans,fullRank = T)
df_dummified <- data.frame(predict(dmy, newdata = df_reordered_omitted_nans))
summary(df_dummified)
str(df_dummified)

#-------------spliting-------------------
#cross-validation in order to avoid overfitting
# I used createDataPartition() which it makes sure the distibution of outcome variable will be similar in test and train
index <- createDataPartition(df$dim_is_requested, p=0.8, list=FALSE)
trainSet <- df[ index,]
testSet <- df[-index,]


str(trainSet)
#-----------------------Feature selection----------------
#1st attempt recursive feature eliminating from Wrapper methods

control <- rfeControl(functions = rfFuncs, method = "repeatedcv", repeats = 3, verbose = FALSE)
outcomeName <- 'dim_is_requested'

str(trainSet)
predictors<-names(trainSet[,-c(2,3)])[!names(trainSet) %in% outcomeName]
predictors <-predictors[1:35]

Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
Loan_Pred_Profile

#----------Boruta--------------
#2nd attempt Boruta
set.seed(100)
boruta.train <- Boruta(dim_is_requested~., data = trainSet, doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

getSelectedAttributes(final.boruta, withTentative = F)
boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)
#---------------Feature selection using Filter methods -------------
# I am using and comparing the salinet features based on their information gain and their X2 values
dim_is_requested.task <- makeClassifTask(data = trainSet, target = "dim_is_requested")
weights<- chi.squared(dim_is_requested~., trainSet)
fv2 = generateFilterValuesData(dim_is_requested.task, method = c("information.gain", "chi.squared"))
plotFilterValues(fv2)




#--------------Feature selcetion---------------------------------
library(party)
cf1 <- cforest(ozone_reading ~ . , data= inputData, control=cforest_unbiased(mtry=2,ntree=50))
#-----------------------Fitting model-----------------------------------------------------------------



#here we can try 2 cateories of discmintaive and generate models
#although it has been suggested that for classification tasks discrimanitve models outperform generatives in general
#another good choice would be ensemle models, such models help in reducing overfitting problems
#To avoid overfitting we can try simpler models like SVM
model_gbm<-caret::train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')# gradient boosting
model_rf<-caret::train (trainSet[,predictors],trainSet[,outcomeName],method='rf') #random forrest
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm') #logstic regression
model_AdaBoost.M1<-train(trainSet[,predictors],trainSet[,outcomeName],method='AdaBoost.M1')
model_svmLinear<-train(trainSet[,predictors],trainSet[,outcomeName],method='svmLinear')


plot(model_gbm)

#----------------Parameter tunning---------------
# we can use tune grid and use grid search to find optimal paramters

modelLookup(model='gbm')
print (model_gbm)
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid)
print(model_gbm)
plot(model_gbm)


#----------variable Importance--------------------
varImp(object=model_gbm)
plot(varImp(object=model_gbm),main="GBM - Variable Importance")

roc_imp2 <- varImp(model_gbm, scale = FALSE)
roc_imp2


#------------Prediction-----------
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])







