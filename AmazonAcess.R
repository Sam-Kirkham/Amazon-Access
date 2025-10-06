library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(GGally)
library(patchwork)
library(glmnet)

############################################################################

# Reading in the Data
testData <- vroom("test.csv")
trainData <- vroom("train.csv")

############################################################################

# Recipe
my_recipe <- recipe(ACTION ~., data=trainData) %>%
  step_mutate_at(vars_I_want_to_mutate, fn = factor) %>% 
  step_other(vars_I_want_other_cat_in, threshold = .001) %>% 
  step_lencode_mixed(vars_I_want_to_target_encode, outcome = vars(target_var))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = testData)