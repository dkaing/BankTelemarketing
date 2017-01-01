# load libraries
library(caret)
library(ggplot2)
library(rattle)
library(pROC)
library(plotROC)
library(ROCR)
library(rpart)
library(randomForest)
library(dplyr)

## read file 
df <- read.csv("~/Downloads/bank/bank-full.csv", sep=";")
df_nodr <- df
df_nodr$duration <- NULL

## data exploratory using random forest
sample <- df[sample(nrow(df), 0.1*nrow(df)),]
rf_sample <- randomForest(y~., data = sample)
rf_samp_vi <- data.frame(varImp(rf_sample))
rf_samp_vi <- data.frame(var = rownames(rf_samp_vi), 
                         value = rf_samp_vi$Overall)
rf_samp_vi <- rf_samp_vi[order(rf_samp_vi$value, decreasing = T), ]
ggplot(rf_samp_vi, aes(reorder(var, value), value))+coord_flip()+
      geom_bar(stat = "identity") +
      xlab("Variable") +
      ylab("Mean Decrease in GINI") +
      ggtitle("Fig 1. VarImp of Sample")+
      theme(text = element_text(size = 12))

# data partitioning
set.seed(1234)

trainPartition <- createDataPartition(df$duration, p = 0.7, list = F)
train_1 <- df[trainPartition, ]
train_2 <- df_nodr[trainPartition,]

test_1 <- df[-trainPartition, ]
test_2 <- df_nodr[-trainPartition,]

## Naive Bayes - training model and record run time

library(e1071)
ptm <- proc.time()
nb1 <- naiveBayes(y~., data = train_1)
nb1t <- proc.time() - ptm

ptm <- proc.time()
nb2 <- naiveBayes(y~., data = train_2)
nb2t <- proc.time() - ptm

nb1_pre <- predict(nb1, test_1)
nb1_cm <- confusionMatrix(nb1_pre, test_1$y)
nb1_overall <- nb1_cm$overall
nb1_byClass <- nb1_cm$byClass

nb2_pre <- predict(nb2, test_2)
nb2_cm <- confusionMatrix(nb2_pre, test_2$y)
nb2_overall <- nb2_cm$overall
nb2_byClass <- nb2_cm$byClass

library(rpart)

ptm <- proc.time()
dt1 <- rpart(y~., method = "class",data = train_1)
pdt1 <- prune(dt1, cp = dt1$cptable[which.min(dt1$cptable[,"xerror"]), "CP"])
dt1t <- proc.time()-ptm

ptm <- proc.time()
dt2 <- rpart(y~., method = "class", data = train_2)
pdt2 <- prune(dt2, cp = dt2$cptable[which.min(dt2$cptable[,"xerror"]), "CP"])
dt2t <- proc.time()-ptm

dt1_pred <- predict(pdt1, test_1, type = "class")
dt1_cm <- confusionMatrix(dt1_pred, test_1$y)
dt1_overall <- dt1_cm$overall
dt1_byClass <- dt1_cm$byClass

dt2_pred <- predict(pdt2, test_2, type = "class")
dt2_cm <- confusionMatrix(dt2_pred, test_2$y)
dt2_overall <- dt2_cm$overall
dt2_byClass <- dt2_cm$byClass

## logistic regression - model & run time 

train_1m <- data.frame(model.matrix(~., data = train_1))
train_1m <- train_1m[,2:ncol(train_1m)]
train_2m <- data.frame(model.matrix(~., data = train_2))
train_2m <- train_2m[,2:ncol(train_2m)]

test_1m <- data.frame(model.matrix(~., data = test_1))
test_1m <- test_1m[,2:ncol(test_1m)]
test_2m <- data.frame(model.matrix(~., data = test_2))
test_2m <- test_2m[,2:ncol(test_2m)]

ptm <- proc.time()
logit1 <- glm(yyes~., binomial(link = "logit"), data = train_1m)
logit1t <- proc.time()-ptm

ptm <- proc.time()
logit2 <- glm(yyes~., binomial(link = "logit"), data = train_2m)
logit2t <- proc.time()-ptm

logit1_pred <- predict(logit1, test_1m, type = "response")
logit1_pred <- ifelse(logit1_pred > 0.5,1,0)
logit1_cm <- confusionMatrix(logit1_pred, test_1m$yyes)
logit1_overall <- logit1_cm$overall
logit1_byClass <- logit1_cm$byClass

logit2_pred <- predict(logit2, test_2m, type = "response")
logit2_pred <- ifelse(logit2_pred > 0.5,1,0)
logit2_cm <- confusionMatrix(logit2_pred, test_2m$yyes)
logit2_overall <- logit2_cm$overall
logit2_byClass <- logit2_cm$byClass

## Random Forest - model & run time

library(randomForest)

ptm <- proc.time()

rf_mod1 <- randomForest(y~., data = train_1)

rf_mod1t <- proc.time() - ptm

ptm <- proc.time()

rf_mod2 <- randomForest(y~., data = train_2)

rf_mod2t <- proc.time() - ptm

pred1 <- predict(rf_mod1, test_1)
pred2 <- predict(rf_mod2, test_2)

rf1_cm <- confusionMatrix(pred1, test_1$y)
rf2_cm <- confusionMatrix(pred2, test_2$y)
rf1_overall <- rf1_cm$overall
rf2_overall <- rf2_cm$overall

rf1_byClass <- rf1_cm$byClass
rf2_byClass <- rf2_cm$byClass


## model by accuracy

mod1 <- data.frame(rbind(nb1_overall, dt1_overall, logit1_overall,  rf1_overall))
mod1$type <- c("Naive\n Bayes", "Decision\n Tree", "Logistic\n Regression", "Random\n Forest")
ggplot(mod1, aes(type, Accuracy)) + 
      geom_bar(stat = "identity")+
      coord_cartesian(ylim = c(0.85,0.95))+
      xlab("Model")+
      theme(text = element_text(size = 12))+
      ggtitle("Figure 2a: Model by Accuracy")

# model by CPU time
time1 <- data.frame(rbind(nb1t, dt1t, logit1t, rf_mod1t))
time1$type <- c("Naive\n Bayes", "Decision\n Tree", "Logistic\n Regression", "Random\n Forest")
ggplot(time1, aes(type, sys.self)) +
      geom_bar(stat = "identity")+
      ylab("CPU Time")+
      xlab("Model")+
      theme(text = element_text(size = 12))+
      ggtitle("Figure 2b: Model by CPU Time")

# ROC 

## Naive Bayes
nb1_pre <- predict(nb1, test_1, type = "raw")
pred <- prediction(nb1_pre[,2], test_1$y)
perf_pred <- performance(pred, measure = "tpr", x.measure = "fpr")
nb_roc <- data.frame(fpr = unlist(perf_pred@x.values), tpr = unlist(perf_pred@y.values))
nb_roc$method <- "Naive Bayes"

## Random Forest
predr1 <- predict(rf_mod1, test_1, type = "prob")
pred_rf1 <- prediction(predr1[,2], test_1$y)
perf_pred_rf1 <- performance(pred_rf1, measure='tpr', x.measure='fpr')
roc_rf1 <- data.frame(fpr=unlist(perf_pred_rf1@x.values), tpr=unlist(perf_pred_rf1@y.values))
roc_rf1$method <- "Random Forest"

## Logistic Regression
logit1_pred <- predict(logit1, test_1m, type = "response")
pred_logis1 <- prediction(logit1_pred, test_1$y)
perf_pred_logis1 <- performance(pred_logis1 , measure='tpr', x.measure='fpr')
roc_lg1 <- data.frame(fpr=unlist(perf_pred_logis1@x.values), tpr=unlist(perf_pred_logis1@y.values))
roc_lg1$method <- "Logistic Regression"

## Decision Tree
dt1r_pred <- predict(pdt1, test_1, type = "prob")
pred_dt <- prediction(dt1r_pred[,2], test_1$y)
perf_pred_dt <- performance(pred_dt, measure='tpr', x.measure='fpr')
roc_dt1 <- data.frame(fpr=unlist(perf_pred_dt@x.values), tpr=unlist(perf_pred_dt@y.values))
roc_dt1$method <- "Decision Tree"

## ROC Plot of all four models
roc <- rbind(nb_roc, roc_rf1, roc_lg1, roc_dt1)

ggplot(roc, aes(fpr, tpr, color = method)) +geom_line()+xlab("False Positive Rate")+
      ylab("True Positive Rate") + theme(legend.title = element_blank(),
                                         text = element_text(size = 12))+
      ggtitle("Figure 3. ROC Plot")


# Variable importance of most accurate model - Random Forest
varimp_rf <- data.frame(varImp(rf_mod1))
varimp_rf <- data.frame(var = rownames(varimp_rf),
                        value = varimp_rf$Overall)
ggplot(varimp_rf, aes(reorder(var,value), value)) +
      geom_bar(stat = "identity")+ coord_flip() +
      ylab("Mean Decrease in GINI") +
      xlab("Variable")+
      theme(text = element_text(size = 12)) +
      ggtitle("Figure 4: Important Variables of Random Forest Model of Entire Dataset")

# Plots of top important variables

df$y <- ifelse(df$y=="yes", 1,0)

## duration
ggplot() +geom_boxplot(data = df, aes(as.factor(y), duration), 
                       fill = "navyblue", color = "darkgray") +
      ylab("Duration") + xlab("Response")+
      theme(text = element_text(size = 12))+
      ggtitle("Figure 5: Duration vs Response")

## month
month_y <- aggregate(y~month, data = df, mean)
levels(month_y$month) <- c("jan", "feb", "mar", 
                           "apr", "may", "jun",
                           "jul", "aug", "sep",
                           "oct", "nov", "dec")
ggplot() +geom_bar(data = month_y, aes(month, y), stat = "identity",
                   fill = "darkred") +
      xlab("Month") + ylab("Success Ratio")+
      theme(text = element_text(size = 12)) +
      ggtitle("Figure 6: Month vs. Success Ratio")

## balance
ggplot() +geom_boxplot(data = df, aes(as.factor(y),balance), 
                       fill = "darkgreen") +
      ylab("Balance") + xlab("Response")+
      theme(text = element_text(size = 12)) +
      ylim(0,5000) +
      ggtitle("Figure 7: Balance vs Success Ratio")

## age
age_y <- aggregate(y~age, data = df, mean)
ggplot() +geom_point(data = age_y, aes(age, y),
                     color = "goldenrod3",
                     size = 3) + 
      scale_x_continuous(breaks = seq(15, 100, by = 10)) +
      xlab("Age") + ylab("Success Ratio")+
      theme(text = element_text(size = 12)) +
      ggtitle("Figure 8: Age vs Ratio")

# Quantifying potential revenue increase based on recommendations
df$month <- as.character(df$month)
subset <- df[df$duration>828 & df$age>=60 | df$age<=20,]
subset <- subset %>% filter(month %in% c("mar", "aug", "nov", "dec"))

improvement <- data.frame(type = c("Original\n Approach", "Recommended\n Approach"), 
                          success_ratio = c(0.12, 0.5))

ggplot(improvement, aes(type, success_ratio)) +
      geom_bar(stat = "identity", fill = "navyblue") + 
      ylab("Success Ratio") +
      xlab("")+
      theme(text = element_text(size = 12)) +
      ggtitle("Figure 9: Increase in Success Ratio with the Recommended Approach")

