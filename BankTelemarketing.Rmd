---
title: "Identifying Bank Telemarketing Market Segmentation Using Classification Algorithms"
author: "by Davin Kaing, Pinkaew Choengtawee, Jiating Chen, Hanbo Li"
date: "January 1, 2017"
output: html_document
---

# Abstract
The objective of our project is to identify the market segment that maximizes business value for bank telemarketing campaign. To do this, we built predictive models using: Logistic Regression, Naive Bayes, Random Forest, and Decision Tree. From our analysis, Random Forest yielded the most accuracy in our prediction. We then chose this model to further explore the features that have the most impact in our model. The top four important variables from this model are: Duration, Month, Balance and Age. Using these important features, we provided business recommendations on how to target the market that maximizes revenue. We then calculated that the market with our recommended approach can potentially increase the success rate by 40%.

# Methodology
The following enumerates our methodology. Each step will be discussed in detail in the subsections below.

1. Data Exploratory Analysis
2. Model Selection
3. Important Variables Selection
4. Business Recommendation
5. Business Value Quantification

# Exploratory Analysis 
In the exploratory analysis, we extracted 30% data of our dataset as a small sample and used Random forest algorithm to detect some likely important independent variables.

From the result, some top variables may be important to the response, such as ‘Duration’, ‘Month’, ‘Balance’, ‘Age’ and ‘Day’ as shown in Figure 1. Although "Duration", which represents the duration of the call, is the most important variable, our team thinks that this can be a confounding variable. Our reasoning is that the duration of the call is driven by the interests of the consumer and we do not think that the duration of the call can drive the interests of the consumer. Further interpretation of this relationship is discussed later in the report.

```{r, echo = FALSE, cache = FALSE, results = "hide", message=FALSE,warning=FALSE}

library(caret)
library(ggplot2)
library(rattle)
library(pROC)
library(plotROC)
library(ROCR)
library(rpart)
library(randomForest)
library(dplyr)

df <- read.csv("~/Downloads/bank/bank-full.csv", sep=";")
df_nodr <- df
df_nodr$duration <- NULL

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
```

# Model Selection

We used four different algorithms: Naive Bayes, Decision Tree, Logistic Regression, and Random Forest to build four predictive models and did a comparative study of them in terms of their accuracy and efficiency. The efficiency was measured by CPU-time in seconds. As shown in Figure 2, Random Forest had the highest accuracy, followed by Logistic Regression, Decision Tree, and Naive Bayes. However, when we examined efficiency, Random Forest also appeared to be the most computationally expensive in comparison to the other models.

```{r, echo = FALSE}

set.seed(1234)

trainPartition <- createDataPartition(df$duration, p = 0.7, list = F)
train_1 <- df[trainPartition, ]
train_2 <- df_nodr[trainPartition,]

test_1 <- df[-trainPartition, ]
test_2 <- df_nodr[-trainPartition,]

## Naive Bayes

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

## logistic regression 

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

## Random Forest
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


## df
mod1 <- data.frame(rbind(nb1_overall, dt1_overall, logit1_overall,  rf1_overall))
mod1$type <- c("Naive\n Bayes", "Decision\n Tree", "Logistic\n Regression", "Random\n Forest")
ggplot(mod1, aes(type, Accuracy)) + 
      geom_bar(stat = "identity")+
      coord_cartesian(ylim = c(0.85,0.95))+
      xlab("Model")+
      theme(text = element_text(size = 12))+
      ggtitle("Figure 2a: Model by Accuracy")


time1 <- data.frame(rbind(nb1t, dt1t, logit1t, rf_mod1t))
time1$type <- c("Naive\n Bayes", "Decision\n Tree", "Logistic\n Regression", "Random\n Forest")
ggplot(time1, aes(type, sys.self)) +
      geom_bar(stat = "identity")+
      ylab("CPU Time")+
      xlab("Model")+
      theme(text = element_text(size = 12))+
      ggtitle("Figure 2b: Model by CPU Time")


```

According to the ROC curve (Figure 3) of 4 different models, Random Forest model has the best performance. With this information, we then used Random Forest model to drive our business recommendation. 

```{r, echo = F}
nb1_pre <- predict(nb1, test_1, type = "raw")
pred <- prediction(nb1_pre[,2], test_1$y)
perf_pred <- performance(pred, measure = "tpr", x.measure = "fpr")
nb_roc <- data.frame(fpr = unlist(perf_pred@x.values), tpr = unlist(perf_pred@y.values))
nb_roc$method <- "Naive Bayes"

predr1 <- predict(rf_mod1, test_1, type = "prob")
pred_rf1 <- prediction(predr1[,2], test_1$y)
perf_pred_rf1 <- performance(pred_rf1, measure='tpr', x.measure='fpr')
roc_rf1 <- data.frame(fpr=unlist(perf_pred_rf1@x.values), tpr=unlist(perf_pred_rf1@y.values))
roc_rf1$method <- "Random Forest"

logit1_pred <- predict(logit1, test_1m, type = "response")
pred_logis1 <- prediction(logit1_pred, test_1$y)
perf_pred_logis1 <- performance(pred_logis1 , measure='tpr', x.measure='fpr')
roc_lg1 <- data.frame(fpr=unlist(perf_pred_logis1@x.values), tpr=unlist(perf_pred_logis1@y.values))
roc_lg1$method <- "Logistic Regression"

dt1r_pred <- predict(pdt1, test_1, type = "prob")
pred_dt <- prediction(dt1r_pred[,2], test_1$y)
perf_pred_dt <- performance(pred_dt, measure='tpr', x.measure='fpr')
roc_dt1 <- data.frame(fpr=unlist(perf_pred_dt@x.values), tpr=unlist(perf_pred_dt@y.values))
roc_dt1$method <- "Decision Tree"

roc <- rbind(nb_roc, roc_rf1, roc_lg1, roc_dt1)

ggplot(roc, aes(fpr, tpr, color = method)) +geom_line()+xlab("False Positive Rate")+
      ylab("True Positive Rate") + theme(legend.title = element_blank(),
                                         text = element_text(size = 12))+
      ggtitle("Figure 3. ROC Plot")

```

# Important Variables Selection

Given that the Random Forest model yields the highest performance, we decided to use this model to drive our business recommendations. To do that, we plotted important variables (from the Random Forest model) according to the mean decrease Gini. From the graph below you can see that the most important variable is duration, followed by month, balance, age and so forth.

```{r, echo = F}
varimp_rf <- data.frame(varImp(rf_mod1))
varimp_rf <- data.frame(var = rownames(varimp_rf),
                        value = varimp_rf$Overall)
ggplot(varimp_rf, aes(reorder(var,value), value)) +
      geom_bar(stat = "identity")+ coord_flip() +
      ylab("Mean Decrease in GINI") +
      xlab("Variable")+
      theme(text = element_text(size = 12)) +
      ggtitle("Figure 4: Important Variables of Random Forest Model of Entire Dataset")

```


```{r, echo = F}
df$y <- ifelse(df$y=="yes", 1,0)

ggplot() +geom_boxplot(data = df, aes(as.factor(y), duration), 
                       fill = "navyblue", color = "darkgray") +
      ylab("Duration") + xlab("Response")+
      theme(text = element_text(size = 12))+
      ggtitle("Figure 5: Duration vs Response")
```




```{r, echo = F}
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

```


```{r, echo = F}
ggplot() +geom_boxplot(data = df, aes(as.factor(y),balance), 
                       fill = "darkgreen") +
      ylab("Balance") + xlab("Response")+
      theme(text = element_text(size = 12)) +
      ylim(0,5000) +
      ggtitle("Figure 7: Balance vs Success Ratio")
```

```{r, echo = F}
age_y <- aggregate(y~age, data = df, mean)
ggplot() +geom_point(data = age_y, aes(age, y),
                     color = "goldenrod3",
                     size = 3) + 
      scale_x_continuous(breaks = seq(15, 100, by = 10)) +
      xlab("Age") + ylab("Success Ratio")+
      theme(text = element_text(size = 12)) +
      ggtitle("Figure 8: Age vs Ratio")

```

# Business Recommendations
After we identified the most important variables, we generated several recommendations from the perspective of the business, which are banks or financial institutions in our case. Below are our recommendations:

1. Train the sale representatives to talk longer than 14 minutes (calculated from our model) with the potential customers.

2. Hire more people or conduct intensive bank telemarketing campaigns in the following months: March, August, November and December.

3. Target the age group below 20 and above 60 years old because they are the people who are more likely to respond “yes”.

One thing worth mentioning is that from the perspective of business, they cannot do anything with potential customers’ balance in their bank account. That’s why we didn’t generate recommendations for this variable.

# Business Value Quantification

Given our recommendations, we are also interested in quantifying the potential increase in business value. We can do this by quantifying the increase in success ratio if our recommendations are implemented (the success ratio is an accurate indicator of business value because for each “yes” response, a quantity of revenue generated. Hence, the success ratio is directly proportional to revenue). To calculate the success ratio given our recommendation, we subset the data according to our business recommendations: the duration is greater than 14 minutes; the months are March, August, November, December; and the age is below 20 and greater than 60. With this subsetted data, we computed the success ratio to be 50% (See Figure 9). As shown in Figure 9, the success ratio increases by approximately 40%. In other words, the revenue can be increased by 4 times if the recommendation is implemented. However, this is under the assumption that the new targeted population, following our recommendation, is representative of the subset population.

```{r, echo = F}

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

```

# Limitation & Future Study

The biggest limitation of our analysis is that the “duration” variable can be a confounding variable. The “duration” variable represents the length of the calls. When we examined the most important variables, “duration” is the most important variable. However, our group thinks that “duration” may not influence the successful responses in the marketing response, rather, we think that the interests of the consumers influence the length of the call, and henceforth, causing the duration to be longer for those who are interested in the product and are willing to respond “yes” to the product. 
With this limitation, we think that future study should delve into the relationship between the duration of the call and the responses. One way to do this is to gather recorded call data (if available), and quantify the consumers’ interest by word choice. We can then use this variable to find a correlation between duration and response. Another possible study is to conduct an experiment where subjects are required to speak to a bank teller for a certain period of time and a survey can be conducted post-experiment to measure their interest in the product.


# Conclusion

Our team used the most accurate classification algorithm to drive the business recommendations. Random Forest provides the most accurate result in comparison to the four classification algorithms. From this model, we then selected the most important variables - Duration, Month, Balance, and Age. We then provided a recommendation to target the market that can generate more revenue. The quantification of this increase is measured by the increased success ratio with the recommendation approach. This increase is calculated to be 40%.

# References

Arvin Fouladifar, E. T. (2016). Market Segmentation for Marketing of Banking Industry Products Constructing a Clustering Model for Bank Pasargad's E-banking Customers Using RFM Technique and K-Means Algorithm. Medwell Journals.

Moro, S., Cortez, P., and Rita, P. (2014) “A Data-Driven Approach to Predict the Success of Bank Telemarketing.” Decision Support Systems. Elsevier. 62:22-31

Mehrotra, A., Agarwal, R. (2009) “Classifying customers on the basis of their attitudes towards telemarketing.” Journal of Targeting, Measurement and Analysis for Marketing. Vol. 17, 3, 171-193

Moro, S., Cortez, P., Laureano, R., (2013) “A Data Mining Approach for Bank Telemarketing Using the rminer Package and R Tool.” ISCTE-IUL, Business Research Unite (BRU-IUL)

Moro S., Laureano R. and Cortez P., (2011) “Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.” In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimaraes, Portugal

Smith, W. R. (1995, December). Product Differentiation and Market Segmentation as Alternative Marketing Strategies. Marketing Management, 4(3), p. 64.




