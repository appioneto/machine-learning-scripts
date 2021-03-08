library(tydr)
library(dplyr)
library(xgboost)
library(caret)
library(ggplot2)
library(corrplot)

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

#check data
head(train_transaction)

#remove V columns - That columns will be transformed by PCA process
train_transaction_v1 <- train_transaction[,c(1:55)]

#check data
head(train_transaction_v1)


#replace NA in factor columns
for (i in colnames(train_transaction_v1)) {
  levels(train_transaction_v1[,i]) <- c(levels(train_transaction_v1[,i]), "-999")
  train_transaction_v1[,i][train_transaction_v1[,i] == ""] <- "-999"
}

#replace NA in numeric columns
for (i in colnames(train_transaction_v1[,c(47:55)])) {
  train_transaction_v1[,i] <- is.numeric(train_transaction_v1[,i])
  train_transaction_v1[,i][train_transaction_v1[,i] == ""] <- -999
}

#check data
head(train_transaction_v1)


#check % of NA
na_percent <- as.data.frame(round(sapply(train_transaction_v1, function(x) 
  (length(x[x == ""]))/nrow(train_transaction_v1)), 4))
na_percent

sapply(train_transaction_v1, function(x) class(x))
sapply(train_transaction_v1, function(x) typeof(x))


###########    label encoding   ##############
library(superml)


#text columns to label encoding
c(5,9,11,16,17)

#drop levels null befor encoding
for (i in colnames(train_transaction_v1[,c(5,9,11,16,17)])){
  if(is.factor(train_transaction_v1[,i]) == TRUE) {
    train_transaction_v1[,i] <-  droplevels(train_transaction_v1[,i])}
}

#create new encoding variables
#train_transaction_v1$ProductCD_lbl
lbl <- LabelEncoder$new()
lbl$fit(train_transaction_v1$ProductCD)
train_transaction_v1$ProductCD_lbl <- lbl$fit_transform(train_transaction_v1$ProductCD)

#train_transaction_v1$card4
lbl <- LabelEncoder$new()
lbl$fit(train_transaction_v1$card4)
train_transaction_v1$card4_lbl <- lbl$fit_transform(train_transaction_v1$card4)

#train_transaction_v1$card6
lbl <- LabelEncoder$new()
lbl$fit(train_transaction_v1$card6)
train_transaction_v1$card6_lbl <- lbl$fit_transform(train_transaction_v1$card6)

#train_transaction_v1$P_emaildomain
lbl <- LabelEncoder$new()
lbl$fit(train_transaction_v1$P_emaildomain)
train_transaction_v1$P_emaildomain_lbl <- lbl$fit_transform(train_transaction_v1$P_emaildomain)

#train_transaction_v1$R_emaildomain
lbl <- LabelEncoder$new()
lbl$fit(train_transaction_v1$R_emaildomain)
train_transaction_v1$R_emaildomain_lbl <- lbl$fit_transform(train_transaction_v1$R_emaildomain)

#remove categorical variables
train_transaction_v2 <- train_transaction_v1[,-c(5,9,11,16,17)]

#visualize head
head(train_transaction_v2)

#Check classes
sapply(train_transaction_v2, function(x) class(x))

#change classes to numeric
for (i in colnames(train_transaction_v2)){
  if(is.numeric(train_transaction_v2[,i]) == FALSE) {
    train_transaction_v2[,i] <-  as.numeric(train_transaction_v2[,i])}
}

#Check classes
sapply(train_transaction_v2, function(x) class(x))

#visualize head
head(train_transaction_v2)



###########    V Columns Correlation    ##############
#V columns dataset
V_Columns <- train_transaction[, c(56:ncol(train_transaction))]

#check data
sapply(V_Columns, function(x) class(x))
sapply(V_Columns, function(x) typeof(x))
head(V_Columns)


#check % of NA
na_percent <- as.data.frame(round(sapply(V_Columns, function(x)
  (sum(is.na(x)))/nrow(V_Columns)), 4))
na_percent

#replace NA in factor columns
for (i in colnames(V_Columns)) {
  levels(V_Columns[,i]) <- c(levels(V_Columns[,i]), "-999")
  V_Columns[,i][V_Columns[,i] == ""] <- "-999"
}

#convert to numeric
for (i in colnames(V_Columns)) {
  V_Columns[,i] <- as.numeric(V_Columns[,i])
}

#check data
sapply(V_Columns, function(x) class(x))
sapply(V_Columns, function(x) typeof(x))
head(V_Columns)

#corrplot
correlation <- cor(V_Columns)

#Correlation matrix
round(correlation, 3)

dim(correlation)

library(psych)
#Bartlett test and KMO
#Input is the correlation matrix
cortest.bartlett(correlation, n=339)
KMO_correlation <- as.matrix(KMO(correlation))


##################   PCA      ####################

#Extraction and number of components
# Scale the data
dataZ <- scale(V_Columns)
head(dataZ)
dim(dataZ)


# Assume the number of components (nfactors) = number of variables, i.e., D = 339,
# Always a unrotated extraction
pc339 <- principal(dataZ, nfactors=339, rotate="none", scores=TRUE)  

#Eigenvalues - Variances of the principal components 
round(pc339$values,3)

#Screeplot - Find the elbow
plot(pc339$values, type = "b", main = "Scree plot for Fraud IEEE dataset",
     xlab = "Number of PC", ylab = "Eigenvalue")   

# Eigenvectors - component loadings
# Each value is the correlation between original variable and component,
# It can be thought as the contribution of each original variable to the component (when squared)
# It summarizes the explained variance too.
pc339$loadings

#Compute the scores
pc10r <- principal(dataZ, nfactors=10, rotate="none", scores=TRUE) 
round(pc10r$scores,3)
mean(pc10r$scores[,1])
sd(pc10r$scores[,1])

#communality represents the amount of variance of the original variable that 
#is summarized by the retained components
round(pc10r$communality,2)

pc10r.scores = (pc10r$scores)
dim(pc10r.scores)
class(pc10r.scores)

#Principals Components boxplot 
boxplot(pc10r.scores[,c(1,2,3)], main = "Three components")
summary(pc10r.scores)

#join PCA 10 with train_transaction_v2
train_transaction_v2 <- cbind(train_transaction_v2, pc10r.scores)

#check data
head(train_transaction_v2)

#############   Split dataset in train / test   ##############
require(Matrix)
require(data.table)
library(xgboost)
library(rminer)

## execute holdout
H=holdout(train_transaction_v2$isFraud,ratio=2/3)
nrow(train_transaction_v2[H$tr,]) #dados de treino
nrow(train_transaction_v2[H$ts,]) #dados de teste
head(train_transaction_v2[H$tr,]) #dados de treino
head(train_transaction_v2[H$ts,]) #dados de teste


#create vector with labels to predict
Y_train = as.data.frame(train_transaction_v2[H$tr,])
Y_train <- Y_train$isFraud
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
Y_test <- Y_test$isFraud
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








#######     Fraud analysis     ########
pred.analiysis <- cbind(train_transaction_v2[H$ts,], pred)

head(pred.analiysis$pred)

library(xlsx)
#Save the new data set in excel format
setwd("C:\\Users\\appio.i.neto\\Documents\\pessoal\\Iscte\\0_Dissertacao\\0_dataset_dissertacao_IEEE-CIS fraud detection\\")
write.csv2(pred.analiysis, file = "fraud_analysis.csv")


head(train_transaction$P_emaildomain)
head(train_transaction$addr2)
library(dplyr)

table(train_transaction$addr2)

train_transaction %>%
  group_by(P_emaildomain)

