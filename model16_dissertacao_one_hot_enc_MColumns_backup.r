library(tydr)
library(dplyr)
library(caret)
library(ggplot2)
library(corrplot)
require(Matrix)
require(data.table)
library(mltools)
library(xgboost)
library(rminer)
library(ROSE)

#Load dataset transaction
train_transaction <- read.csv2("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\train_transaction.csv", sep = ",")

##Load dataset identity
#train_identity <- read.csv2("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\train_identity.csv", sep = ",")

#check dimensions
dim(train_transaction)
#dim(train_identity)

#Statistical summary transaction
summary(train_transaction)

#class os variables transaction
as.data.frame(sapply(train_transaction, function(x) class(x)))

#First observations
head(train_transaction)

#############   Cleaning data - replace NA   ##############
# 
# #check % of NA
# na_percent <- as.data.frame(round(sapply(train_transaction, function(x) 
#   (length(x[x == ""]))/nrow(train_transaction)), 4))
# na_percent

#class of variables transaction
as.data.frame(sapply(train_transaction, function(x) class(x)))

#type of variables transaction
as.data.frame(sapply(train_transaction, function(x) typeof(x)))

#convert to numeric all columns - convert factors to number of level
for (i in colnames(train_transaction)) {
  train_transaction[,i] <- as.numeric(train_transaction[,i])
}

# #check % of NA
# na_percent <- as.data.frame(round(sapply(train_transaction, function(x) 
#   (sum(x[is.na(x)]))/nrow(train_transaction)), 4))
# na_percent
# 
#replace NA in numeric columns
for (i in colnames(train_transaction)) {
  if (is.numeric(train_transaction[,i]) == TRUE){
    train_transaction[,i][is.na(train_transaction[,i])] <- -999
  }
}

#Visualize head
head(train_transaction)
colnames(train_transaction[,47:55])
sapply(train_transaction[,47:55], function(x) length(x[x == -999]))


#replace -999 in M columns
for (i in colnames(train_transaction[,47:55])) {
    train_transaction[,i][train_transaction[,i] == -999] <- -1
}

#Check Data
sapply(train_transaction[,47:55], function(x) table(x))


# #replace -999 by -1 (other option)
# train_transaction_v2[,c(47:55)] <- sapply(train_transaction_v2[,c(47:55)], function(x) ifelse(x == -999, -1, x))


###########    Feature Engineering    ##############


# create day column - the first tansaction date is 86400 (total of seconds in 1 day)
# Lets assume 86400 being day one
train_transaction$transaction_day <- as.integer(train_transaction$TransactionDT / 86400)

# create week column
train_transaction$transaction_week <- as.integer((train_transaction$transaction_day / 7) + 0.4)

# create week column
train_transaction$transaction_month <- as.integer((train_transaction$transaction_week / 4) + 0.4)

#check data
head(train_transaction[, c("transaction_month", "transaction_week", "transaction_day")])
tail(train_transaction[, c("transaction_month", "transaction_week", "transaction_day")])


#remove V columns
train_transaction_v2 <- train_transaction[,c(2:55, 395:ncol(train_transaction))]
colnames(train_transaction_v2)
head(train_transaction_v2)


#One Hot Encoding M Columns
#Convert columns to factor
dummy <- dummyVars(" ~ .", data=train_transaction_v2)
train_transaction_v2 <- data.frame(predict(dummy, newdata = train_transaction_v2))

#Check Data
head(train_transaction_v2)

###############   PCA V Columns   #######################

library(psych)

#Extraction and number of components
# Scale the data over 100 first v columns
dataZ <- scale(train_transaction[, c(56:156)])
# dataZ
# summary(dataZ)
# ncol(dataZ)

#Let's extract a 10 component solution and Compute the scores
pc10 <- principal(dataZ, nfactors=10, rotate="none", scores=TRUE)

#Screeplot - Find the elbow
plot(pc10$values, type = "b", main = "Scree plot",
     xlab = "Number of PC", ylab = "Eigenvalue") 

pc10$loadings
round(pc10$communality,2)
round(pc10$scores,3)
mean(pc10$scores[,1])
sd(pc10$scores[,1])

#Add 3 PC to dataset
train_transaction_v2$pc1 <- pc10$scores[,1]
train_transaction_v2$pc2 <- pc10$scores[,2]
train_transaction_v2$pc3 <- pc10$scores[,3]


#Extraction and number of components
# Scale the data over 101 to 200 v columns
dataZ <- scale(train_transaction[, c(157:257)])
# dataZ
# summary(dataZ)
# ncol(dataZ)

#Let's extract a 10 component solution and Compute the scores
pc10_2 <- principal(dataZ, nfactors=10, rotate="none", scores=TRUE)

#Screeplot - Find the elbow
plot(pc10_2$values, type = "b", main = "Scree plot",
     xlab = "Number of PC", ylab = "Eigenvalue") 

pc10_2$loadings
round(pc10_2$communality,2)
round(pc10_2$scores,3)
mean(pc10_2$scores[,1])
sd(pc10_2$scores[,1])

#Add 3 PC to dataset
train_transaction_v2$pc4 <- pc10_2$scores[,1]
train_transaction_v2$pc5 <- pc10_2$scores[,2]
train_transaction_v2$pc6 <- pc10_2$scores[,3]
train_transaction_v2$pc7 <- pc10_2$scores[,4]
train_transaction_v2$pc8 <- pc10_2$scores[,5]
train_transaction_v2$pc9 <- pc10_2$scores[,6]



#Extraction and number of components
# Scale the data over plus 201 v columns
dataZ <- scale(train_transaction[, c(258:395)])
# dataZ
# summary(dataZ)
# ncol(dataZ)

#Let's extract a 10 component solution and Compute the scores
pc10_3 <- principal(dataZ, nfactors=10, rotate="none", scores=TRUE)

#Screeplot - Find the elbow
plot(pc10_3$values, type = "b", main = "Scree plot",
     xlab = "Number of PC", ylab = "Eigenvalue") 

pc10_3$loadings
round(pc10_3$communality,2)
round(pc10_3$scores,3)
mean(pc10_3$scores[,1])
sd(pc10_3$scores[,1])

#Add 4 PC to dataset
train_transaction_v2$pc10 <- pc10_3$scores[,1]
train_transaction_v2$pc11 <- pc10_3$scores[,2]
train_transaction_v2$pc12 <- pc10_3$scores[,3]
train_transaction_v2$pc13 <- pc10_3$scores[,4]

# scale funtion got smaller score
# #Normalize columns except Isfraud and M columns
# train_transaction_v2[,c(2:45)] <- sapply(train_transaction_v2[,c(2:45)], function(x) x <- scale(x))
# #Check Data
# head(train_transaction_v2)


# #### Tentative create frquency column for card columns
# colnames(train_transaction[,c(6:11)])
# sapply(train_transaction[,c(6:11)], function(x) table(x))
# 
# for (i in colnames(train_transaction[,c(6:11)])){
#   temp <- as.data.frame(table(train_transaction[,i]))
#   temp$Var1 <- as.double(temp$Var1) 
#   paste("train_transaction$", i, "_freq",sep = "") <-
#     left_join(as.integer(train_transaction[,i]), temp, 
#               by = c(i, as.integer(temp$Var1)))
# }
# head(ave(train_transaction, 
#          train_transaction$card1, FUN = length))
# head(transform(train_transaction, freq.loc = ave(train_transaction$card1, )), card1, FUN = length)))
# head(train_transaction[train_transaction$card1 == 13926,c(6)])
# 
# tapply(train_transaction$card1, function(x) length(x))
# 
# for (i in colnames(train_transaction[,c(6:11)])){
#   temp <- as.data.frame(table(train_transaction[,i]))
#   temp$Var1 <- as.double(temp$Var1) 
#   paste("train_transaction$", i, "_freq",sep = "") <-
#     ifelse(train_transaction$card1 == temp$card1, temp$Freq, 0)
#               by = c(i, as.integer(temp$Var1)))
# }
# class(temp$card1)
# class(train_transaction[,i])
# 
# head(train_transaction)
# head(temp)
# tail(as.data.frame(table(train_transaction$card1)), 50)
# colnames(temp) <- c("card1", "Freq")
# 
# head(merge(x = as.data.frame(train_transaction$card1), y = temp, by = "card1"))

# #Normaliza D Columns 
# train_transaction$D1n <- train_transaction$D1 - train_transaction$transaction_day
# train_transaction$D2n <- train_transaction$D2 - train_transaction$transaction_day
# train_transaction$D3n <- train_transaction$D3 - train_transaction$transaction_day
# train_transaction$D5n <- train_transaction$D5 - train_transaction$transaction_day
# train_transaction$D9n <- train_transaction$D9 - train_transaction$transaction_day
# 
# # add column with card1 grouping
# q <- data.frame(quantile(train_transaction$card1, c(.15, .3, .45, .6, .75, .9)))
# 
# train_transaction$card1_group <- 0
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 <= 3747.85, as.integer(1), as.integer(0))
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 > 3747.85 & train_transaction$card1 <= 6951.00
#          , as.integer(2), train_transaction$card1_group)
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 > 6951.00 & train_transaction$card1 <= 9175.00
#          , as.integer(3), train_transaction$card1_group)
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 > 9175.00 & train_transaction$card1 <= 11313.00
#          , as.integer(4), train_transaction$card1_group)
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 > 11313.00 & train_transaction$card1 <= 14184.00
#          , as.integer(5), train_transaction$card1_group)
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 > 14184.00 & train_transaction$card1 <= 16582.10
#          , as.integer(6), train_transaction$card1_group)
# 
# train_transaction$card1_group <- 
#   ifelse(train_transaction$card1 > 16582.10
#          , as.integer(7), train_transaction$card1_group)
# 
# #check data
# table(train_transaction$card1_group)
# 
# 
# 
# # add column with card2 grouping
# q2 <- data.frame(quantile(train_transaction$card2, c(.2, .4, .6, .8)))
# 
# train_transaction$card2_group <- 0
# 
# train_transaction$card2_group <- 
#   ifelse(train_transaction$card2_group <= 75, as.integer(1), train_transaction$card2_group)
# 
# train_transaction$card2_group <- 
#   ifelse(train_transaction$card2 > 75 & train_transaction$card2 <= 222
#          , as.integer(2), train_transaction$card2_group)
# 
# train_transaction$card2_group <- 
#   ifelse(train_transaction$card2 > 222 & train_transaction$card2 <= 334
#          , as.integer(3), train_transaction$card2_group)
# 
# train_transaction$card2_group <- 
#   ifelse(train_transaction$card2 > 334 & train_transaction$card2 <= 435
#          , as.integer(4), train_transaction$card2_group)
# 
# train_transaction$card2_group <- 
#   ifelse(train_transaction$card2 > 435
#          , as.integer(5), train_transaction$card2_group)
# 
# #check data
# table(train_transaction$card2_group)


# # Product_weigthFraud got decrease AUC
# # analyse on ProductCD
# train_transaction %>%
#   group_by(ProductCD, isFraud) %>%
#   summarize(count = n())
# 
# # create Product_weigthFraud
# train_transaction$Product_weigthFraud <- 
#   ifelse(train_transaction$ProductCD == 1, round(as.double(1.000),3), 0)
# 
# train_transaction$Product_weigthFraud <- 
#          ifelse(train_transaction$ProductCD == 2, round(as.double(0.283),3), train_transaction$Product_weigthFraud)
# 
# train_transaction$Product_weigthFraud <- 
#          ifelse(train_transaction$ProductCD == 3, round(as.double(0.181),3), train_transaction$Product_weigthFraud)
# 
# train_transaction$Product_weigthFraud <- 
#          ifelse(train_transaction$ProductCD == 4, round(as.double(0.400),3), train_transaction$Product_weigthFraud)
# 
# train_transaction$Product_weigthFraud <-
#          ifelse(train_transaction$ProductCD == 5, round(as.double(0.000),3), train_transaction$Product_weigthFraud)
# 
# #check data
# table(train_transaction$Product_weigthFraud)
# table(train_transaction$ProductCD)

###########################################################################

# #add feature dist1 / dist2
# train_transaction$relation_dist <- train_transaction$dist2 / train_transaction$dist1
# 
# head(train_transaction$range_relation_dist, 30)
# typeof(train_transaction$range_relation_dist)
# 
# #label seq(0,1,.1)
# train_transaction$range_relation_dist <- 
#   ifelse (trin_transaction$relation_dist <= 1, 
#     as.double(cut(train_transaction$relation_dist[train_transaction$relation_dist <= 1], seq(0,1,.1), right = FALSE, labels = c(1:10))),
#     0
#     )
# 
# head(train_transaction$relation_dist
#       [train_transaction$relation_dist <= 10 ], 20)
# table(train_transaction$range_relation_dist)
# 
# 
# train_transaction$range_relation_dist <- 
#   ifelse (train_transaction$relation_dist > 1 && train_transaction$relation_dist <= 10, 
#           as.double(cut(train_transaction$relation_dist[train_transaction$relation_dist > 1 |train_transaction$relation_dist <= 10],
#                         seq(2,10,1), right = FALSE, labels = c(10:20))),
#           train_transaction$range_relation_dist
#   )
# 
# print(seq(2,10,1))
# 
# train_transaction$range_relation_dist <- 
#   ifelse (train_transaction$relation_dist > 1 |train_transaction$relation_dist <= 10, 
#           as.double(cut(train_transaction$relation_dist[train_transaction$relation_dist > 1 |train_transaction$relation_dist <= 10],
#                         seq(10,100,10), right = FALSE, labels = c(11:21))),
#           train_transaction$range_relation_dist
#   )


#############   Split dataset in train / test   ##############

## execute holdout
H=holdout(train_transaction_v2[,1],ratio=2/3)
nrow(train_transaction_v2[H$tr,]) #dados de treino
nrow(train_transaction_v2[H$ts,]) #dados de teste
head(train_transaction_v2[H$tr,]) #dados de treino
head(train_transaction_v2[H$ts,]) #dados de teste

#create vector with labels to predict
Y_train = as.data.frame(train_transaction_v2[H$tr,])
sum(Y_train$isFraud)

#Apply ROSE technique to balance classes
Y_train.rose <- ROSE(isFraud ~ ., data = Y_train, seed = 1)$data
table(Y_train.rose$isFraud)
#combine NoFraud from Y_train with Fraud from ROSE
Y_train.rose2 <- rbind(Y_train, sample_n(Y_train.rose[Y_train.rose$isFraud == 1,], 50000))
table(Y_train.rose2$isFraud)

#20726 / 400693
#5.172% of Fraud

#27681 / 407693
#6.789% of Fraud

#63681 / 443693
#14.352% of Fraud

# #Apply Undersampling   
# Y_train.under <- ovun.sample(isFraud ~ ., data = Y_train, method = "under", N = 27374, seed = 1)$data
# table(Y_train.under$isFraud)
# 
# #Apply oversampling   
# Y_train.over <- ovun.sample(isFraud ~ ., data = Y_train, method = "over", N = 410000)$data
# table(Y_train.over$isFraud)


##################################################

#create training dataset
X_train <- as.matrix(Y_train.rose2[,-c(1,2)])

#convert to dgCMatrix
X_train = as(X_train, "dgCMatrix")

#Y_train label
Y_train <- Y_train.rose2[, 2]
colnames(Y_train.rose2)
#Check Data
nrow(Y_train)
nrow(X_train)
head(Y_train)
head(X_train)
table(Y_train)


#Setting up test dataset
#create vector with labels to validate perdiction
Y_test = as.data.frame(train_transaction_v2[H$ts,])

#create test dataset
X_test <- as.matrix(train_transaction_v2[H$ts,-c(1,2)])

#convert to dgCMatrix
X_test = as(X_test, "dgCMatrix")

#Y_test label
Y_test <- Y_test[, 2]

nrow(Y_test)
nrow(X_test)
head(Y_test)
head(X_test)
table(Y_test)
table(Y_train)

##############   train model     ################



bst <- xgboost(data = X_train, label = Y_train, max.depth = 16,
               eta = 1, nthread = 2, nrounds = 100,
               objective = "binary:logistic", verbose = 1,
               early_stopping_rounds = 3, eval_metric = "auc",
               tree_method = "hist", scale_pos_weight = (sum(Y_train)/nrow(X_train)))


##############   Test model     ################

#Make predictions
pred <- predict(bst, X_test)

# #Export to excel
# pred_analisys <- train_transaction_v2[H$ts,]
# head(pred_analisys)
# head(round(as.double(pred), 2),50)
# 
# setwd("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\")
# write.csv2(pred_analisys, file = "fraud_transact.csv")
# write.csv2(round(as.double(pred), 3), file = "fraud_transact_pred.csv")

########   Interval analysis   ###############
KS_analysis <- as.data.frame(cbind(Y_test, pred))

#remove range_predictions
remove(range_predictions)

#Create range_predictions
range_predictions <- data.frame("RangeOfPredictions" = c("> 0.0 and < 0.01"),
                                "SumOfFraud" = sum(KS_analysis[KS_analysis$pred < .01, "Y_test"]),
                                "SumOf_NO_Fraud" = nrow(KS_analysis[KS_analysis$pred < .01,])-
                                  sum(KS_analysis[KS_analysis$pred < .01,"Y_test"])
)

#change class from factor to char - RangeOfPredictions column
range_predictions$RangeOfPredictions <- as.character(range_predictions$RangeOfPredictions)
class(range_predictions$RangeOfPredictions)


#add ">= 0.01 and < 0.02"
range_predictions <- rbind(range_predictions,
                           list(">= 0.01 and < 0.02", 
                                sum(KS_analysis[KS_analysis$pred >= .01 & KS_analysis$pred < .02,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .01 & KS_analysis$pred < .02,])-
                                  sum(KS_analysis[KS_analysis$pred >= .01 & KS_analysis$pred < .02,"Y_test"]))
)

#add ">= 0.02 and < 0.03"
range_predictions <- rbind(range_predictions,
                           list(">= 0.02 and < 0.03", 
                                sum(KS_analysis[KS_analysis$pred >= .02 & KS_analysis$pred < .03,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .02 & KS_analysis$pred < .03,])-
                                  sum(KS_analysis[KS_analysis$pred >= .02 & KS_analysis$pred < .03,"Y_test"]))
)

#add ">= 0.03 and < 0.04"
range_predictions <- rbind(range_predictions,
                           list(">= 0.03 and < 0.04", 
                                sum(KS_analysis[KS_analysis$pred >= .03 & KS_analysis$pred < .04,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .03 & KS_analysis$pred < .04,])-
                                  sum(KS_analysis[KS_analysis$pred >= .03 & KS_analysis$pred < .04,"Y_test"]))
)

#add ">= 0.04 and < 0.05"
range_predictions <- rbind(range_predictions,
                           list(">= 0.04 and < 0.05", 
                                sum(KS_analysis[KS_analysis$pred >= .04 & KS_analysis$pred < .05,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .04 & KS_analysis$pred < .05,])-
                                  sum(KS_analysis[KS_analysis$pred >= .04 & KS_analysis$pred < .05,"Y_test"]))
)

#add ">= 0.05 and < 0.06"
range_predictions <- rbind(range_predictions,
                           list(">= 0.05 and < 0.06", 
                                sum(KS_analysis[KS_analysis$pred >= .05 & KS_analysis$pred < .06,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .05 & KS_analysis$pred < .06,])-
                                  sum(KS_analysis[KS_analysis$pred >= .05 & KS_analysis$pred < .06,"Y_test"]))
)

#add ">= 0.06 and < 0.07"
range_predictions <- rbind(range_predictions,
                           list(">= 0.06 and < 0.07", 
                                sum(KS_analysis[KS_analysis$pred >= .06 & KS_analysis$pred < .07,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .06 & KS_analysis$pred < .07,])-
                                  sum(KS_analysis[KS_analysis$pred >= .06 & KS_analysis$pred < .07,"Y_test"]))
)

#add ">= 0.07 and < 0.08"
range_predictions <- rbind(range_predictions,
                           list(">= 0.07 and < 0.08", 
                                sum(KS_analysis[KS_analysis$pred >= .07 & KS_analysis$pred < .08,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .07 & KS_analysis$pred < .08,])-
                                  sum(KS_analysis[KS_analysis$pred >= .07 & KS_analysis$pred < .08,"Y_test"]))
)

#add ">= 0.08 and < 0.09"
range_predictions <- rbind(range_predictions,
                           list(">= 0.08 and < 0.09", 
                                sum(KS_analysis[KS_analysis$pred >= .08 & KS_analysis$pred < .09,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .08 & KS_analysis$pred < .09,])-
                                  sum(KS_analysis[KS_analysis$pred >= .08 & KS_analysis$pred < .09,"Y_test"]))
)

#add ">= 0.09 and < 0.1
range_predictions <- rbind(range_predictions,
                           list(">= 0.09 and < 0.1", 
                                sum(KS_analysis[KS_analysis$pred >= .09 & KS_analysis$pred < .1,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .09 & KS_analysis$pred < .1,])-
                                  sum(KS_analysis[KS_analysis$pred >= .09 & KS_analysis$pred < .1,"Y_test"]))
)

#add ">= 0.1 and < 0.2"
range_predictions <- rbind(range_predictions,
                           list(">= 0.1 and < 0.2", 
                                sum(KS_analysis[KS_analysis$pred >= .1 & KS_analysis$pred < .2,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .1 & KS_analysis$pred < .2,])-
                                  sum(KS_analysis[KS_analysis$pred >= .1 & KS_analysis$pred < .2,"Y_test"]))
)

#add ">= 0.2 and < 0.3"
range_predictions <- rbind(range_predictions,
                           list(">= 0.2 and < 0.3", 
                                sum(KS_analysis[KS_analysis$pred >= .2 & KS_analysis$pred < .3,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .2 & KS_analysis$pred < .3,])-
                                  sum(KS_analysis[KS_analysis$pred >= .2 & KS_analysis$pred < .3,"Y_test"]))
)

#add ">= 0.3 and < 0.4"
range_predictions <- rbind(range_predictions,
                           list(">= 0.3 and < 0.4", 
                                sum(KS_analysis[KS_analysis$pred >= .3 & KS_analysis$pred < .4,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .3 & KS_analysis$pred < .4,])-
                                  sum(KS_analysis[KS_analysis$pred >= .3 & KS_analysis$pred < .4,"Y_test"]))
)

#add ">= 0.4 and < 0.5"
range_predictions <- rbind(range_predictions,
                           list(">= 0.4 and < 0.5", 
                                sum(KS_analysis[KS_analysis$pred >= .4 & KS_analysis$pred < .5,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .4 & KS_analysis$pred < .5,])-
                                  sum(KS_analysis[KS_analysis$pred >= .4 & KS_analysis$pred < .5,"Y_test"]))
)

#add ">= 0.5 and < 0.6"
range_predictions <- rbind(range_predictions,
                           list(">= 0.5 and < 0.6", 
                                sum(KS_analysis[KS_analysis$pred >= .5 & KS_analysis$pred < .6,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .5 & KS_analysis$pred < .6,])-
                                  sum(KS_analysis[KS_analysis$pred >= .5 & KS_analysis$pred < .6,"Y_test"]))
)

#add ">= 0.6 and < 0.7"
range_predictions <- rbind(range_predictions,
                           list(">= 0.6 and < 0.7", 
                                sum(KS_analysis[KS_analysis$pred >= .6 & KS_analysis$pred < .7,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .6 & KS_analysis$pred < .7,])-
                                  sum(KS_analysis[KS_analysis$pred >= .6 & KS_analysis$pred < .7,"Y_test"]))
)

#add ">= 0.7 and < 0.8"
range_predictions <- rbind(range_predictions,
                           list(">= 0.7 and < 0.8", 
                                sum(KS_analysis[KS_analysis$pred >= .7 & KS_analysis$pred < .8,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .7 & KS_analysis$pred < .8,])-
                                  sum(KS_analysis[KS_analysis$pred >= .7 & KS_analysis$pred < .8,"Y_test"]))
)

#add ">= 0.8 and < 0.9"
range_predictions <- rbind(range_predictions,
                           list(">= 0.8 and < 0.9", 
                                sum(KS_analysis[KS_analysis$pred >= .8 & KS_analysis$pred < .9,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .8 & KS_analysis$pred < .9,])-
                                  sum(KS_analysis[KS_analysis$pred >= .8 & KS_analysis$pred < .9,"Y_test"]))
)

#add ">= 0.9"
range_predictions <- rbind(range_predictions,
                           list(">= 0.9 and < 9999", 
                                sum(KS_analysis[KS_analysis$pred >= .9,"Y_test"]),
                                nrow(KS_analysis[KS_analysis$pred >= .9,])-
                                  sum(KS_analysis[KS_analysis$pred >= .9,"Y_test"]))
)


##############   Test validation     ################

#set Scores results
#prediction <- as.numeric(pred > 0.5)
#prediction <- as.numeric(pred >= 0.01)
prediction <- ifelse(pred < 0.04, 0, 1)



#fraud distribution
table(prediction)
table(Y_test)


#Make AUC plot
library(AUC)
roc_curve <- roc(pred, as.factor(Y_test))
sens <- sensitivity(pred, as.factor(Y_test))
spec <- specificity(pred, as.factor(Y_test))
accu <- accuracy(pred, as.factor(Y_test))

#AUC result
auc(roc_curve)
auc(sens)
auc(spec)
auc(accu)

#line graph
plot(roc_curve, y = NULL, type = "l",
     add = FALSE, min = 0, max = 1)

#error rate
err <- mean(prediction != Y_test)
print(paste("test-error=", err))

#Make a confusion matrix
library(caret)
confusionMatrix(as.factor(prediction), as.factor(Y_test))

#############   Importance features analysis   ###################

# Importance Matrix
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)


xgboost_new_features <- xgb.create.features(bst, X_train)
colnames(xgboost_new_features)

################   Deep Learning Dreams    ##############
a = as.vector(X_test[1,])
a
image(matrix(as.integer(c(((a-min(a))/(max(a)-0))*(255-0)+min(a), 0)), nrow=6, ncol=9))

v=62
u=765

((765-0)/(1246-0))*(255-0)+0
((1246-0)/(1246-0))*(255-0)+0
