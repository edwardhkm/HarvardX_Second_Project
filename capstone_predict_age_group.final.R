###############################################################
# Create training set, validation set (final hold-out test set)
###############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(corrr)) install.packages("corrr", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# Loading
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(lubridate)
library(corrr)
library(ggcorrplot)
library(xgboost)
library(reshape2)
library(dplyr)
library(gridExtra)
library(matrixStats)
library(randomForest)

dl <- tempfile()

## Read csv file 
download.file("https://dphi.s3.ap-south-1.amazonaws.com/dataset/train_age_dataset.csv", dl)
my_data_org <- read.csv(dl, stringsAsFactors=FALSE)
glimpse(my_data_org)

# Test NA of data
anyNA(my_data_org)
names(my_data_org)

# Display general structure of our data
str(my_data_org)

# Show basic stat of our data
summary(my_data_org)

# Ensure no duplicated userId
n_occur <- data.frame(table(my_data_org$userId))
n_occur[n_occur$Freq > 1,]

# Remove temp file as placeholder for the zip file
rm(dl)

# Test with smaller set 10,000 obs
set.seed(1, sample.kind="Rounding")
index <- sample(1:nrow(my_data_org), 10000, replace = FALSE)
my_data_shortlist <- my_data_org[index,]

# Remove user name and user Id
n_col <- 25
remove_cols <- c('Unnamed..0','userId')
my_data <- my_data_shortlist[, !(colnames(my_data_shortlist)%in%remove_cols), drop=FALSE]


############################################################################
# Data set 2 for nearZeroVar
############################################################################
library(matrixStats)
sds <- colSds(as.matrix(my_data))
qplot(sds)
hist(sds)
str(sds)
View(sds)

nzv <- nearZeroVar(my_data)
col_index <- setdiff(1:ncol(my_data), nzv)
ncol_nzv <- length(col_index)

my_data_nzv <- my_data

# Change target variable to factor type
my_data_nzv$age_group <- as.factor(my_data_nzv$age_group)
my_data_nzv$tier <- as.factor(my_data_nzv$tier)
my_data_nzv$gender <- as.factor(my_data_nzv$gender)

my_data_nzv <- my_data_nzv[ ,col_index]

# We take 90% training set and 10% test set
test_index_nzv <- createDataPartition(y = my_data_nzv$age_group, times = 1, p = 0.1, list = FALSE)
train_set_nzv <- my_data_nzv[-test_index_nzv, ]
test_set_nzv <- my_data_nzv[test_index_nzv, ]

# Define predictors and response variables in the training set
train_x_nzv <- train_set_nzv[, -ncol_nzv]
train_y_nzv <- train_set_nzv[,ncol_nzv]

# Define predictors and response variables in the test set
test_x_nzv <- test_set_nzv[, -ncol_nzv]
test_y_nzv <- test_set_nzv[, ncol_nzv]

# Setup parameters for KNN
tune_grid <- data.frame(k = seq(10, 100, 10))
control <- trainControl(method = "cv", number = 10, p = .9)

train_knn_nzv <- train(age_group ~ ., data = train_set_nzv,
                   method = "knn",
                   tuneGrid = tune_grid,
                   trControl = control)

plot(train_knn_nzv)
best_tune_knn_nzv <- train_knn_nzv$bestTune

y_hat_knn_nzv <- predict(train_knn_nzv, test_set_nzv, type = "raw")

cm_knn_nzv <- confusionMatrix(y_hat_knn_nzv, test_y_nzv)$overall["Accuracy"]
print(cm_knn_nzv)


############################################################################
# Data Visualization
#
############################################################################
mydata.cor = round(cor(my_data), 3)
head(mydata.cor)
ggcorrplot(mydata.cor)

p1 <- qplot(log(my_data_org$weekends_trails_watched_per_day), bins = 30)
p2 <- qplot(log(my_data_org$weekdays_trails_watched_per_day), bins = 30)
p3 <- qplot(log(my_data_org$slot4_trails_watched_per_day), bins = 30)
p4 <- qplot(log(my_data_org$slot3_trails_watched_per_day), bins = 30)
p5 <- qplot(log(my_data_org$slot2_trails_watched_per_day), bins = 30)
p6 <- qplot(log(my_data_org$slot1_trails_watched_per_day), bins = 30)

grid.arrange(p1, p2, p3, p4, p5, p6, ncol=2)

############################################################################
# Random forest Model
#
############################################################################

# Change target variable to factor type
my_data$age_group <- as.factor(my_data$age_group)
my_data$tier <- as.factor(my_data$tier)
my_data$gender <- as.factor(my_data$gender)

# We take 90% training set and 10% test set
test_index <- createDataPartition(y = my_data$age_group, times = 1, p = 0.1, list = FALSE)
train_set <- my_data[-test_index, ]
test_set <- my_data[test_index, ]

# Define predictors and response variables in the training set
train_x <- train_set[, -n_col]
train_y <- train_set[,n_col]

# Define predictors and response variables in the test set
test_x <- test_set[, -n_col]
test_y <- test_set[, n_col]

# Setup parameters for Random Forest
control <- trainControl(method="cv", number=5, search ="grid")
grid <- data.frame(mtry=c(1,10,50,100,500))

# Run the model
train_rf <- train(age_group ~ ., 
                  data = train_set, 
                  method="rf",
                  trControl=control,
                  tuneGrid = grid,
                  metric = "Accuracy",
                  importance = TRUE)

# Print the results
print(train_rf)

# Plot training result
ggplot(train_rf)
best_mtry <- train_rf$bestTune 
print(best_mtry)
paste("The best_mtry is ", best_mtry, sep="")

fit_rf <- randomForest(train_x, train_y,
                       minNode = train_rf$bestTune$mtry)
plot(fit_rf)
y_hat_rf <- predict(fit_rf, test_set)
cm_rf <- confusionMatrix(y_hat_rf, test_y)
cm_rf$overall["Accuracy"]


############################################################################
# KNN model
#
############################################################################

# The normalization function is created
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }

# make a copy of our sample data
my_data <- my_data_shortlist[, !(colnames(my_data_shortlist)%in%remove_cols), drop=FALSE]

# Normalized features for KNN
select_features <- seq(1:(n_col-1))
my_data_norm <- as.data.frame(lapply(my_data[, select_features], nor))

# Generated new normalized data set.  Outcome is appended back.
my_data_norm <- data.frame(my_data_norm, my_data$age_group)
col_names <- colnames(my_data_norm[,select_features])
colnames(my_data_norm) <- c(col_names, "age_group")

my_data_norm$age_group <- as.factor(my_data_norm$age_group)
str(my_data_norm)

# Create normalized test set and training set
set.seed(1, sample.kind="Rounding")
train_set <- my_data_norm[-test_index, ]
test_set <- my_data_norm[test_index, ]

# Define predictors and response variables in the training set
train_x <- train_set[, -n_col]
train_y <- train_set[,n_col]

# Define predictors and response variables in the test set
test_x <- test_set[, -n_col]
test_y <- test_set[, n_col]

tune_grid <- data.frame(k = seq(10, 100, 10))
control <- trainControl(method = "cv", number = 10, p = .9)

train_knn <- train(age_group ~ ., data = train_set,
                   method = "knn",
                   tuneGrid = tune_grid,
                   trControl = control)

plot(train_knn)
best_tune_knn <- train_knn$bestTune

y_hat_knn <- predict(train_knn, test_set, type = "raw")

cm_knn <- confusionMatrix(y_hat_knn, test_y)$overall["Accuracy"]
print(cm_knn)

############################################################################
# XGBoost model
#
############################################################################

train_set <- my_data[-test_index, ]
test_set <- my_data[test_index, ]

# Define predictors and response variables in the training set
train_x <- data.matrix(train_set[, -n_col])
train_y <- train_set[,n_col]

# Define predictors and response variables in the test set
test_x <- data.matrix(test_set[, -n_col])
test_y <- test_set[, n_col]

# Define final training and testing sets for XGboost
# XGBoost package can use matrix dat.  so we'll use matrix command to conver training and
# Test set into matrix.
set.seed(1, sample.kind="Rounding")
xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

#  Define watchlist
watchlist <- list(train=xgb_train, test=xgb_test)

# Try fine tune number of rounds
N_rounds <- seq(10, 5000, 250)
preds6 <- sapply(N_rounds, function(n){
  first_model <- xgb.train(data = xgb_train, max.depth = 6, watchlist=watchlist, nrounds = n, verbose = 0, lambda = 1)
  pred_y <- predict(first_model, as.matrix(as.integer(test_y)))
  pred_y_hat <- round(pred_y)
  
  u <- union(pred_y_hat, test_y)
  t <- table(factor(pred_y_hat, u), factor(test_y, u))
  cm <- confusionMatrix(t)
  cm_accuracy <- cm$overall["Accuracy"]
  result <- c(6, n, cm_accuracy)
})
preds6 <- t(preds6)
colnames(preds6) <- c('depths', 'rounds', 'Accuarcy')
preds6

model_result <- c("Random Forest", "KNN", "XGBoost")
model_accuracy <-c(0.6849452, 0.6370887, 0.6251246)
df <- data.frame(model_result, model_accuracy)
df

