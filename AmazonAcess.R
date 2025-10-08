library(tidyverse)
library(tidymodels)
library(embed)
library(ggmosaic)
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
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
bakedData <- bake(prep, new_data = testData)
ncol(bakedData)

############################################################################

# EDA
trainData <- trainData %>%
  mutate(ACTION = as.factor(ACTION)) %>% 
  mutate(RESOURCE = as.factor(RESOURCE)) %>% 
  mutate(MGR_ID = as.factor(MGR_ID)) %>% 
  mutate(ROLE_ROLLUP_1 = as.factor(ROLE_ROLLUP_1)) %>% 
  mutate(ROLE_ROLLUP_2 = as.factor(ROLE_ROLLUP_2)) %>% 
  mutate(ROLE_DEPTNAME = as.factor(ROLE_DEPTNAME)) %>% 
  mutate(ROLE_TITLE = as.factor(ROLE_TITLE)) %>% 
  mutate(ROLE_FAMILY_DESC = as.factor(ROLE_FAMILY_DESC)) %>% 
  mutate(ROLE_FAMILY = as.factor(ROLE_FAMILY)) %>% 
  mutate(ROLE_CODE = as.factor(ROLE_CODE))

# step_mutate_at(all_numeric_predictors(), fn = factor)


ggplot(data = trainData) +
  geom_mosaic(aes(x = product(RESOURCE, MGR_ID), fill = ACTION))

