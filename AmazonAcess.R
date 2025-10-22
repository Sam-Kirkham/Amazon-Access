library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(DataExplorer)
library(GGally)
library(patchwork)
library(glmnet)
library(discrim)

############################################################################

# Reading in the Data
testData <- vroom("test.csv")
trainData <- vroom("train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

############################################################################

# EDA
# trainData <- trainData %>%
#   mutate(ACTION = as.factor(ACTION)) %>%
#   mutate(RESOURCE = as.factor(RESOURCE)) %>%
#   mutate(MGR_ID = as.factor(MGR_ID)) %>%
#   mutate(ROLE_ROLLUP_1 = as.factor(ROLE_ROLLUP_1)) %>%
#   mutate(ROLE_ROLLUP_2 = as.factor(ROLE_ROLLUP_2)) %>%
#   mutate(ROLE_DEPTNAME = as.factor(ROLE_DEPTNAME)) %>%
#   mutate(ROLE_TITLE = as.factor(ROLE_TITLE)) %>%
#   mutate(ROLE_FAMILY_DESC = as.factor(ROLE_FAMILY_DESC)) %>%
#   mutate(ROLE_FAMILY = as.factor(ROLE_FAMILY)) %>%
#   mutate(ROLE_CODE = as.factor(ROLE_CODE))
# 
# # step_mutate_at(all_numeric_predictors(), fn = factor)
# 
# 
# ggplot(data = trainData) +
#   geom_mosaic(aes(x = product(RESOURCE, MGR_ID), fill = ACTION))

############################################################################

# Recipe

my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())


prep <- prep(my_recipe)
bakedData <- bake(prep, new_data = testData)

############################################################################

# Logistic Regression
logRegModel <- logistic_reg() %>% 
  set_engine("glm")

logreg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=trainData)

amazon_predictions <- predict(logreg_wf,
                              new_data=testData,
                              type= 'prob')

############################################################################

# Penalized Logistic Regression

penalized_logistic <- logistic_reg(mixture=tune(), penalty=tune()) %>%
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penalized_logistic)
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(trainData, v = 5, repeats=1)

CV_results <- amazon_workflow %>%
    tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

amazon_predictions <- final_wf %>%
  predict(new_data = testData, type= 'prob')


############################################################################

# Binary Random Forests
RF_Binary_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

RF_Binary_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(RF_Binary_mod)
tuning_grid <- grid_regular(mtry(range = c(1, 9)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(trainData, v = 5, repeats=1)

CV_results <- RF_Binary_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc, f_meas, sens, recall, precision, accuracy))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_wf <- RF_Binary_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

amazon_predictions <- final_wf %>%
  predict(new_data = testData, type= 'prob')

############################################################################

# K-Nearest Neighbors

knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

tuning_grid <- grid_regular(neighbors(),
                levels = 5)

folds <- vfold_cv(trainData, v = 5, repeats=1)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

amazon_predictions <- final_wf %>%
  predict(new_data = testData, type = 'prob')

############################################################################

# Naive Bayes

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(trainData, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

amazon_predictions <- final_wf %>%
  predict(new_data = testData, type= 'prob')

############################################################################

submission <- amazon_predictions %>%
  bind_cols(testData) %>%
    select(id, .pred_1) %>%
    rename(ACTION = .pred_1) %>%
    rename(ID = id)
  
vroom_write(submission, file = "./AmazonPreds.csv", delim = ",")

# FileZIlla: sftp://stat-u01.byu.edu
# Run a batch on the Stat Servers,R CMD BATCH --no-save --no-restore AmazonAcess.R &

