library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(DataExplorer)
library(GGally)
library(patchwork)
library(glmnet)
library(discrim)
library(kernlab)
library(themis)

# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")
# reticulate::install_python()
# keras::install_keras()
# library(nnet)

############################################################################

# Reading in the Data
testData <- vroom("test.csv")
trainData <- vroom("train.csv") %>%
  mutate(ACTION = factor(ACTION))

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
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_nominal_predictors()) %>%
  step_pca(all_predictors(), threshold= 0.68)


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
            metrics=metric_set(roc_auc))

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
                            levels = 10)

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

# Neural Network

nn_recipe <- recipe(formula = ACTION ~ ., data = trainData) %>%
  # update_role(id, new_role = "id") %>%
  step_mutate(color = as.factor(color)) %>%  # Turn color to factor
  step_dummy(all_nominal_predictors()) %>%   # Then dummy encode color
  step_range(all_numeric_predictors(), min = 0, max = 1) # scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%  
  set_engine("keras") %>%  
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range = c(1, 20)),
                            levels = 10)

nn_folds <- vfold_cv(trainData, v = 5, repeats=1)

tuned_nn <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

tuned_nn %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line()

final_nn <- nn_wf %>%
  finalize_workflow(select_best(tuned_nn, "accuracy")) %>%
  fit(data = trainData)

amazon_predictions <- predict(final_nn, testData) %>%
  bind_cols(testData %>% select(ACTION))

############################################################################

# PCA

recipe(formula=, data=) %>%
  ... %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=) 
#Threshold is between 0 and 1

prep <- prep(my_recipe)
bakedData <- bake(prep, new_data = testData)


############################################################################

# Support Vector Machines

amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.99) %>%
  step_downsample(ACTION)

svm_rbf_model <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_model <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_linear_model <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(svm_linear_model)

final_wf <- svm_wf %>%
  fit(data=trainData)

amazon_predictions <- final_wf %>%
  predict(new_data = testData, type= 'prob')


############################################################################

# Imbalanced Data
my_recipe <- recipe(ACTION ~., data=trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  # step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_nominal_predictors()) %>%
  # only one of those
  # step_smote(all_outcomes(), neighbors = 4)
  # step_upsample(ACTION)
  step_downsample(ACTION)


prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = testData)


############################################################################
submission <- amazon_predictions %>%
  bind_cols(testData) %>%
    select(id, .pred_1) %>%
    rename(ACTION = .pred_1) %>%
    rename(ID = id)
  
vroom_write(submission, file = "./AmazonPreds.csv", delim = ",")

# FileZIlla: sftp://stat-u01.byu.edu
# Run a batch on the Stat Servers,R CMD BATCH --no-save --no-restore AmazonAcess.R &

