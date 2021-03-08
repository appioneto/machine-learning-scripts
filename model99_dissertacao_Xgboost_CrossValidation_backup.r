library(tydr)
library(dplyr)
library(caret)
library(ggplot2)
library(corrplot)
require(Matrix)
require(data.table)
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

###########    Feature Engineering    ##############


# create day column - the first tansaction date is 86400 (total of seconds in 1 day)
# Lets assume 86400 being day one
train_transaction$transaction_day <- as.integer(train_transaction$TransactionDT / 86400)
#check data
head(train_transaction$transaction_day)
tail(train_transaction$transaction_day)


# create week column
train_transaction$transaction_week <- as.integer((train_transaction$transaction_day / 7) + 0.4)
#check data
head(train_transaction[, c("transaction_week", "transaction_day")])
tail(train_transaction[, c("transaction_week", "transaction_day")])

# create week column
train_transaction$transaction_month <- as.integer((train_transaction$transaction_week / 4) + 0.4)
#check data
head(train_transaction[, c("transaction_month", "transaction_week", "transaction_day")])
tail(train_transaction[, c("transaction_month", "transaction_week", "transaction_day")])


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

colnames(train_transaction)
ncol(train_transaction)
#remove V columns
train_transaction_v2 <- train_transaction[,c(1:55, 395:ncol(train_transaction))]
colnames(train_transaction_v2)
head(train_transaction_v2)
# #remove card1, card 2, D1, D2, D3, D5, D9
# train_transaction_v2 <- train_transaction_v2[, -c(6,7,18, 19, 20, 22, 24)]

## execute holdout
H=holdout(train_transaction_v2[,1],ratio=2/3)
nrow(train_transaction_v2[H$tr,]) #dados de treino
nrow(train_transaction_v2[H$ts,]) #dados de teste
head(train_transaction_v2[H$tr,]) #dados de treino
head(train_transaction_v2[H$ts,]) #dados de teste


#create vector with labels to predict
Y_train = as.data.frame(train_transaction_v2[H$tr,])
sum(Y_train$isFraud)
head(Y_train)

#Apply ROSE technique to balance classes
Y_train.rose <- ROSE(isFraud ~ ., data = Y_train, seed = 1)$data
table(Y_train.rose$isFraud)
head(Y_train.rose[Y_train.rose$isFraud == 1,])
#combine Ytrain with Fraud from ROSE
Y_train.rose2 <- rbind(Y_train, sample_n(Y_train.rose[Y_train.rose$isFraud == 1,], 10000))
table(Y_train.rose2$isFraud)



#Apply Undersampling   
Y_train.under <- ovun.sample(isFraud ~ ., data = Y_train, method = "under", N = 27374, seed = 1)$data
table(Y_train.under$isFraud)

#Apply oversampling   
Y_train.over <- ovun.sample(isFraud ~ ., data = Y_train, method = "over", N = 410000)$data
table(Y_train.over$isFraud)

# library(DMwR)
# Y_train.smote <- SMOTE(isFraud ~., as.factor(Y_train), perc.over = 100, perc.under = 200)
# 
# library(imbalance)
# Y_train.mwmote <- mwmote(Y_train, numInstances = 100, classAttr = "isFraud")

##################################################
#Y label
Y_train <- Y_train.rose2[, 2]

#create training dataset
X_train <- as.matrix(Y_train.rose2[,-c(1,2)])


#convert to dgCMatrix
X_train = as(X_train, "dgCMatrix")

nrow(Y_train)
nrow(X_train)
head(Y_train)
head(X_train)
tail(X_train)
table(Y_train)


##############   train model     ################
#testar
xgb.create.features()


bst <- xgboost(data = X_train, label = Y_train, max.depth = 16,
               eta = 1, nthread = 2, nrounds = 100,
               objective = "binary:logistic", verbose = 1,
               early_stopping_rounds = 3, #eval_metric =  list(metric = "error", value = err),
               eval_metric = "auc",
               tree_method = "hist", scale_pos_weight = (sum(Y_train)/nrow(X_train)))

##############   Test model     ################

#create vector with labels to predict
Y_test = as.data.frame(train_transaction_v2[H$ts,2])
sum(Y_test)


#create training dataset
X_test <- as.matrix(scale(train_transaction_v2[H$ts,-c(1,2)]))


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
prediction <- as.numeric(pred >= 0.01)
prediction <- ifelse(pred < 0.03, 0, 1)



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

######### #make a test 
xgb.create.features()

##############   Improve cost of error to algorithm     ################



# Revenue function given expected revenue of 3 for true negatives,
# and -10 for false negatives. Revenue/cost of 0 for true and false positives.
getProfit <- function(Y_test, prediction) {
  tp_count <- sum(Y_test == 1 & Y_test == prediction)
  tn_count <- sum(Y_test == 0 & Y_test == prediction)
  fp_count <- sum(Y_test == 0 & Y_test != prediction)
  fn_count <- sum(Y_test == 1 & Y_test != prediction)
  profit <- tn_count * 3 - fn_count * 10
  
  return(profit)
}

threshold <- 0.23

# get profit if we would predict every observation as "non-returning"
getBenchmarkProfit <- function(obs) { # predict non-returning for everyone
  n <- length(obs)
  getProfit(obs, rep(0, times = n))
}

# get the lift of our predictions over the benchmark profit;
# defined as getProfit/getBenchmarkProfit
getLift <- function(probs, labels, tresh) {
  pred_profit <- as.numeric(getProfit(obs = labels,
                                      pred = probs,
                                      treshold = tresh))
  naive_profit <- as.numeric(getBenchmarkProfit(labels))
  profit_lift <- pred_profit/naive_profit
  return(profit_lift)
}


library(xgboost)

# train features contains the training data, train_label the dependent
# variable; test features contains the test data, test_label the dependent
# variable of the test set.
dtrain <- xgb.DMatrix(X_train, label = Y_train)
dtest <- xgb.DMatrix(X_test, label = Y_test)

xgb_params <- list(objective = "binary:logistic",
                   eta = 1,
                   max_depth = 16,
                   colsample_bytree = 1,
                   subsample = 0.75,
                   min_child_weight = 1)

thresholds <- seq(0.1, 0.69, by = 0.01)
performance <- vector(length = 60)

for(i in 1:9) {
  # define the function every iteration to use a new threshold
  xgb.getLift <- function(prediction, dtrain) {
    labels <- getinfo(dtrain, "label")
    lift <- getLift(preds, labels, thresholds[i])
    return(list(metric = "Lift", value = lift))
  }
  set.seed(512)
  # train the model again using the current iteration's threshold 
  xgb_fit <- xgb.cv(params = xgb_params, data = dtrain, nfold = 5,
                    feval = xgb.getLift, maximize = TRUE, 
                    nrounds = 100, early_stopping_rounds = 5,
                    verbose = TRUE)
  # store the results
  performance[i] <- as.data.frame(xgb_fit$evaluation_log)[
    xgb_fit$best_iteration,4]
}

# print the optimal threshold
thresholds[which.max(performance)]




bst <- xgboost(data = X_train, label = Y_train, max.depth = 16,
               eta = 1, nthread = 2, nrounds = 100,
               objective = "binary:logistic", verbose = 1,
               early_stopping_rounds = 3, eval_metric = "auc",
               tree_method = "hist", scale_pos_weight = (sum(Y_train)/nrow(X_train)))




############   Deep Learning Dreams - transform rows into vectors (image in deep learning)   ##############
a = as.vector(X_test[1,])
a
image(matrix(as.integer(c(((a-min(a))/(max(a)-0))*(255-0)+min(a), 0)), nrow=6, ncol=9))

v=62
u=765

((765-0)/(1246-0))*(255-0)+0
((1246-0)/(1246-0))*(255-0)+0
