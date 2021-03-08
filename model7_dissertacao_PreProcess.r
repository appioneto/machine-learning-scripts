library(tydr)
library(dplyr)
library(caret)
library(ggplot2)
library(corrplot)
require(Matrix)
require(data.table)
library(xgboost)
library(rminer)

#Load dataset transaction
train_transaction <- read.csv2("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\train_transaction.csv", sep = ",")

##Load dataset identity
train_identity <- read.csv2("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\train_identity.csv", sep = ",")

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

#replace NA in numeric columns
for (i in colnames(train_transaction)) {
  if (is.numeric(train_transaction[,i]) == TRUE){
    train_transaction[,i][is.na(train_transaction[,i])] <- -999
  }
}

#check % of NA
na_percent <- as.data.frame(round(sapply(train_transaction, function(x) 
  (sum(x[is.na(x)]))/nrow(train_transaction)), 4))
na_percent

#Visualize head
head(train_transaction)

###########    Feature Engineering    ##############

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


dim(train_transaction)
table(train_transaction$ProductCD)

#############   Split dataset in train / test   ##############

#remove V columns
train_transaction_v2 <- train_transaction[,c(1:55)]


## execute holdout
H=holdout(train_transaction_v2[,1],ratio=2/3)
nrow(train_transaction_v2[H$tr,]) #dados de treino
nrow(train_transaction_v2[H$ts,]) #dados de teste
head(train_transaction_v2[H$tr,]) #dados de treino
head(train_transaction_v2[H$ts,]) #dados de teste



#create vector with labels to predict
Y_train = as.data.frame(train_transaction_v2[H$tr,])
Y_train <- Y_train[, 2]
#create training dataset
X_train <- as.matrix(train_transaction_v2[H$tr,-c(1,2)])

#convert to dgCMatrix
X_train = as(X_train, "dgCMatrix")

dim(Y_train)
nrow(X_train)
head(Y_train)
head(X_train)
table(Y_train)




##############   train model     ################

bst <- xgboost(data = X_train, label = Y_train, max.depth = 16,
               eta = 1, nthread = 2, nrounds = 100,
               objective = "binary:logistic", verbose = 1,
               early_stopping_rounds = 3, eval_metric = "auc",
               tree_method = "hist", scale_pos_weight = (sum(Y_train)/nrow(X_train)))

##############   Test model     ################

#create vector with labels to predict
Y_test = as.data.frame(train_transaction_v2[H$ts,])
Y_test <- Y_test[, 2]
#create training dataset
X_test <- as.matrix(train_transaction_v2[H$ts,-c(1,2)])

#convert to dgCMatrix
X_test = as(X_test, "dgCMatrix")


nrow(Y_test)
nrow(X_test)
head(Y_test)
head(X_test)
table(Y_test)
table(Y_train)


#Make predictions
pred <- predict(bst, X_test)

# #Export to excel
# pred_analisys <- head(cbind(train_transaction_v2[H$ts,-c(1)], round(pred, 4)))
# head(pred_analisys)
# 
# setwd("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\")
# write.csv2(fraud_transact, file = "fraud_transact.csv")



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
prediction <- as.numeric(pred >= 0.03)


#fraud distribution
table(prediction)
table(Y_test)


#Make AUC plot
library(AUC)
roc_curve <- roc(as.factor(prediction), as.factor(Y_test))
sens <- sensitivity(as.factor(prediction), as.factor(Y_test))
spec <- specificity(as.factor(prediction), as.factor(Y_test))
accu <- accuracy(as.factor(prediction), as.factor(Y_test))

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




################   Deep Learning Dreams    ##############
a = as.vector(X_test[1,])
a
image(matrix(as.integer(c(((a-min(a))/(max(a)-0))*(255-0)+min(a), 0)), nrow=6, ncol=9))

v=62
u=765

((765-0)/(1246-0))*(255-0)+0
((1246-0)/(1246-0))*(255-0)+0
