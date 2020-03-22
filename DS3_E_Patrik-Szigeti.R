# Load libraries
library(data.table)
library(skimr)
library(caret)
library(ggplot2)
library(pROC)
library(ROCR)
library(caTools)

# Set working directory
setwd("C:/Users/szige/Desktop/CEU/2019-2020 Winter/Data Science 3 - ML2/Assignments/")

### Exploratory Data Analysis
# Read in the train dataset
raw_data_train <- fread("data/online_news_popularity/train.csv")
raw_data_train$set <- "train"

raw_data_test <- fread("data/online_news_popularity/test.csv")
raw_data_test$set <- "test"
raw_data_test$is_popular <- NA

data <- rbind(raw_data_train, raw_data_test)

### Data Cleaning
data$is_popular <- factor(data$is_popular, labels = c("not_popular", "popular"))

data$data_channel_is_lifestyle <- factor(data$data_channel_is_lifestyle)
data$data_channel_is_entertainment <- factor(data$data_channel_is_entertainment)
data$data_channel_is_bus <- factor(data$data_channel_is_bus)
data$data_channel_is_socmed <- factor(data$data_channel_is_socmed)
data$data_channel_is_tech <- factor(data$data_channel_is_tech)
data$data_channel_is_world <- factor(data$data_channel_is_world)

data$weekday_is_monday <- factor(data$weekday_is_monday)
data$weekday_is_tuesday <- factor(data$weekday_is_tuesday)
data$weekday_is_wednesday <- factor(data$weekday_is_wednesday)
data$weekday_is_thursday <- factor(data$weekday_is_thursday)
data$weekday_is_friday <- factor(data$weekday_is_friday)
data$weekday_is_saturday <- factor(data$weekday_is_saturday)
data$weekday_is_sunday <- factor(data$weekday_is_sunday)
data$is_weekend <- NULL

data_train <- subset(data, set == "train")
data_train$set <- NULL

data_test <- subset(data, set == "test")
data_test$set <- NULL
data_test$is_popular <- NULL

rm(raw_data_test, raw_data_train, data)

### Predictions

# Linear model prediction
train_control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,
  savePredictions = TRUE
)

set.seed(1234)
linear_model <- train(
  is_popular ~ .,
  data = data_train,
  preProcess = c("center", "scale"),
  method = "glm",
  trControl = train_control
)

lm_roc <- roc(
  predictor = predict(
    linear_model, 
    data_train, 
    type = "prob", 
    decision.values = TRUE)$popular, 
  response = data_train$is_popular
)
lm_roc
plot(lm_roc)

linear_results <- data.table(
  article_id = data_test$article_id,
  score = predict(
    linear_model,
    data_test,
    type = "prob",
    decision.values = TRUE)$popular
)

write.csv(linear_results, "data/online_news_popularity/predictions/linear_results.csv", row.names = FALSE)

# Random forest prediction
tune_grid <- expand.grid(
  .mtry = c(2, 3, 5, 8),
  .splitrule = "gini",
  .min.node.size = c(5, 10)
)

set.seed(1234)
rf_model <- train(
  is_popular ~ .,
  method = "ranger",
  data = data_train,
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "impurity"
)

predicted_probabilities <- predict(rf_model, newdata = data_train, type = "prob")
rocr_prediction <- prediction(predicted_probabilities[["popular"]], data_train[["is_popular"]])
plot(performance(rocr_prediction, "tpr", "fpr"), col = "black")
abline(a = 0, b = 1, col = "#8AB63F")

rf_results <- data.table(
  article_id = data_test$article_id,
  score = predict(
    rf_model, 
    newdata = data_test, 
    type = "prob", 
    decision.values = TRUE)$popular
)

write.csv(rf_results, "data/online_news_popularity/predictions/rf_results.csv", row.names = FALSE)

### Gradient boosting prediction
# GBM model 1
gbm_grid_1 <- expand.grid(
  n.trees = 1000, 
  interaction.depth = 1, 
  shrinkage = 0.001,
  n.minobsinnode = 1
)

set.seed(1234)
gbm_model_1 <- train(
  is_popular ~ .,
  method = "gbm",
  data = data_train,
  trControl = train_control,
  tuneGrid = gbm_grid_1,
  verbose = FALSE
)

gbm_roc_1 <- roc(
  predictor = predict(
    gbm_model_1, 
    data_train, 
    type = "prob", 
    decision.values = TRUE)$popular, 
  response = data_train$is_popular
)

# GBM model 2
gbm_grid_2 <- expand.grid(
  n.trees = 500, 
  interaction.depth = 5, 
  shrinkage = 0.1,
  n.minobsinnode = 5
)

set.seed(1234)
gbm_model_2 <- train(
  is_popular ~ .,
  method = "gbm",
  data = data_train,
  trControl = train_control,
  tuneGrid = gbm_grid_2,
  verbose = FALSE
)

gbm_roc_2 <- roc(
  predictor = predict(
    gbm_model_2, 
    data_train, 
    type = "prob", 
    decision.values = TRUE)$popular, 
  response = data_train$is_popular
)

# GBM model 3
gbm_grid_3 <- expand.grid(
  n.trees = 500, 
  interaction.depth = 10, 
  shrinkage = 0.1,
  n.minobsinnode = 5
)

set.seed(1234)
gbm_model_3 <- train(
  is_popular ~ .,
  method = "gbm",
  data = data_train,
  trControl = train_control,
  tuneGrid = gbm_grid_3,
  verbose = FALSE
)

gbm_roc_3 <- roc(
  predictor = predict(
    gbm_model_3, 
    data_train, 
    type = "prob", 
    decision.values = TRUE)$popular, 
  response = data_train$is_popular
)

# Select the best GBM model
plot(gbm_roc_1)
plot(gbm_roc_2, add = TRUE, col = "blue")
plot(gbm_roc_3, add = TRUE, col = "green")

print(paste("AUC of gbm_roc_1:", gbm_roc_1$auc))
print(paste("AUC of gbm_roc_2:", gbm_roc_2$auc))
print(paste("AUC of gbm_roc_3:", gbm_roc_3$auc))

# Save the results of the best GBM model
gbm_results <- data.table(
  article_id = data_test$article_id,
  score = predict(
    gbm_model_3, 
    newdata = data_test, 
    type = "prob", 
    decision.values = TRUE)$popular
)

write.csv(gbm_results, "data/online_news_popularity/predictions/gbm_results.csv", row.names = FALSE)

###. H2O
library(h2o)
h2o.no_progress() # suppress progress bars in the outcome
h2o.init(max_mem_size = "6g")

data_train_h2o <- as.h2o(data_train)
data_test_h2o <- as.h2o(data_test)

y <- "is_popular"
X <- setdiff(names(data_train_h2o), y)

# Random forest
rf_grid_h2o <- h2o.grid(
  x = X, y = y, 
  training_frame = data_train_h2o, 
  algorithm = "randomForest",
  nfolds = 5,
  seed = 1234,
  hyper_params = list(
    ntrees = 500,
    mtries = 5
  ),
  keep_cross_validation_predictions = TRUE
)

rf_model_h2o <- h2o.getModel(
  h2o.getGrid(rf_grid_h2o@grid_id)@model_ids[[1]]
)

# GLM model
glm_model_h2o <- h2o.glm(
  X, y,
  training_frame = data_train_h2o,
  family = "binomial",
  alpha = 1,
  lambda_search = TRUE,
  seed = 1234,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)

# GBM model
gbm_model_h2o <- h2o.gbm(
  X, y,
  training_frame = data_train_h2o,
  ntrees = 500, 
  max_depth = 10, 
  learn_rate = 0.1, 
  seed = 1234,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)

# Deep learning model
dl_model_h2o <- h2o.deeplearning(
  X, y,
  training_frame = data_train_h2o,
  hidden = c(32, 8),
  seed = 1234,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)

validation_performances <- list(
  "rf" = h2o.auc(h2o.performance(rf_model_h2o, newdata = data_train_h2o)),
  "glm" = h2o.auc(h2o.performance(glm_model_h2o, newdata = data_train_h2o)),
  "gbm" = h2o.auc(h2o.performance(gbm_model_h2o, newdata = data_train_h2o)),
  "dl" = h2o.auc(h2o.performance(dl_model_h2o, newdata = data_train_h2o))
)

validation_performances

ensemble_model_dl <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train_h2o,
  metalearner_algorithm = "deeplearning",
  seed = 1234,
  base_models = list(
    rf_model_h2o,
    glm_model_h2o, 
    gbm_model_h2o,
    dl_model_h2o
  )
)

print(h2o.auc(h2o.performance(ensemble_model_dl, newdata = data_train_h2o)))

ensemble_model_dl_results <- h2o.cbind(
  h2o.predict(
    ensemble_model_dl, 
    newdata = data_test_h2o, 
    type = "prob", 
    decision.values = TRUE),
  data_test_h2o$article_id
)[, c("article_id", "popular")]

names(ensemble_model_dl_results)[2] = c("score")

h2o.exportFile(ensemble_model_dl_results, "data/online_news_popularity/predictions/ensemble_model_dl_results.csv", sep = ",")

### Neural network prediction
# caret
tune_grid_nnet_1 <- expand.grid(
  size = c(3, 5, 7, 10, 15),
  decay = c(0.1, 0.5, 1, 1.5, 2, 2.5, 5)
)

set.seed(1234)
nnet_model <- train(
  is_popular ~ .,
  method = "nnet",
  data = data_train,
  trControl = train_control,
  tuneGrid = tune_grid_nnet_1,
  preProcess = c("center", "scale", "pca"),
  metric = "ROC",
  trace = FALSE
)
nnet_model

nnet_prediction <- prediction(
  predict.train(
    nnet_model, 
    newdata = data_train,
    type = "prob")$popular,
  data_train$is_popular
)
performance(nnet_prediction, measure = "auc")@y.values[[1]]

nnet_results_1 <- data.table(
  article_id = data_test$article_id,
  score = predict(
    nnet_model, 
    newdata = data_test, 
    type = "prob", 
    decision.values = TRUE)$popular
)

write.csv(nnet_results_1, "data/online_news_popularity/predictions/nnet_results_1.csv", row.names = FALSE)
