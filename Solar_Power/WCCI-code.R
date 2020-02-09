# Starting Code -----------------------------------------------------
## Clear memory
rm(list=ls(all=TRUE))

## Set working directory
setwd("~/Artigos/WCCI/")

BaseDir       <- getwd()
ResultsDir    <- paste(BaseDir, "Results", sep="/")
DataDir       <- paste(BaseDir, "Data", sep="/")

## Load libraries
library(ggplot2)
library(dplyr)
library(Metrics)
library(mlbench)
library(caret)
library(caretEnsemble)
library(e1071)
library(readxl)
library(forecast)
library(quantregForest)

source('Feature_Engineering.R')
source('Plot.R')
source('Plot2.R')

# Data treatment ----------------------------------------------------
setwd(DataDir)

## Load data
dataset <- read_excel("wcci.xlsx", col_types = c("date", 
                                              "numeric", "numeric", "numeric", "numeric"))

## Difference to create output Power
dataset$Power <- dataset$Meter_principal - dataset$Power_total

## Removing Meter_Principal and Power_Total
dataset <- dataset[,-c(2,3)]

## Removing NAs
dataset <- dataset[complete.cases(dataset),]

## Selecting 3 days (April 14, 15 and 16)

dataset <- dataset[23720:24150,]

## ACF and PACF plots to determine lag
acf(dataset$Power, main = "")
pacf(dataset$Power, main = "")
auto.arima(dataset$Power)

## Lag as inputs
lag <- 1

# x1 = temperature; x2 = radiation
## Creating inputs and applying lags
data <- data.frame(dataset$Power[(lag+1):(dim(dataset)[1])],
                  dataset$Power[(lag:(dim(dataset)[1]-lag))],
                  dataset$Temperature[(lag+1):(dim(dataset)[1])],
                  dataset$Temperature[(lag:(dim(dataset)[1]-lag))],
                  dataset$Radiation[(lag+1):(dim(dataset)[1])],
                  dataset$Radiation[(lag:(dim(dataset)[1]-lag))])
colnames(data) <- c('y','y(t-1)','x1','x1(t-1)','x2','x2(t-1)')

## creating features engineering for y and y(t-1)
features <- FE(data[,c(1,2)])
colnames(features[[1]]) <- c('meanstra.1','meanstra.2',
                             'sdtra.1','sdtra.2',
                             'skewtra.1','skewtra.2',
                             'diftra.1','diftra.2',
                             'expo2.1','expo2.2',
                             'expo3.1','expo3.2',
                             'flog.1','flog.2',
                             'min.1','min.2',
                             'max.1','max.2')

data <- data.frame(data,features[[1]])


## Training and Test sets
n <- dim(data)[1]
cut <- 0.7 * n

train <- data[1:cut,]
test <- tail(data,n-cut)

x_train <- train[,-1]
y_train <- train[,1]

x_test <- test[,-1]
y_test <- test[,1]

# saving the treated data
setwd(ResultsDir)
save.image("treated-data.RData")

# Training and Predictions ------------------------------------------
# setwd(ResultsDir)
# load("treateddata.RData")

set.seed(1234)

control <- trainControl(method = "timeslice",
                        initialWindow = 0.7*dim(train)[1],
                        horizon = 0.2*dim(train)[1],
                        fixedWindow = FALSE,
                        allowParallel = TRUE,
                        savePredictions = 'final',
                        verboseIter = FALSE)

model.list <- c('svmLinear2', 'earth', 'brnn', 'lm')
  
# c('gaussprRadial','knn', 'rf', 'svmRadial')

## svmRadial
## svmLinear2, earth, brnn, cubist, lm sao bons demais

pp.list <- c('pca','corr')

model <- list()
pred.train <- NA
pred.test  <- NA
Pred <- matrix(ncol = length(model.list)*length(pp.list), nrow = n)
colnames(Pred) <- c('pca-svr','pca-mars','pca-brnn','pca-lm',
                    'corr-svr','corr-mars','corr-brnn','corr-lm')

metrics.train <- matrix(nrow = length(model.list)*length(pp.list), ncol = 4)
metrics.test <- matrix(nrow = length(model.list)*length(pp.list), ncol = 4)
colnames(metrics.train) <- c('k','SMAPE','RRMSE','R2')
rownames(metrics.train) <- c('pca-svr','pca-mars','pca-brnn','pca-lm',
                             'corr-svr','corr-mars','corr-brnn','corr-lm')

colnames(metrics.test) <- c('k','SMAPE','RRMSE','R2')
rownames(metrics.test) <- c('pca-svr','pca-mars','pca-brnn','pca-lm',
                            'corr-svr','corr-mars','corr-brnn','corr-lm')

k <- 1

for (i in seq(pp.list)) {
  for (j in seq(model.list)) {
    ## model training
    model[[k]] <- train(y~., data = train,
                        method = model.list[j],
                        trControl = control,
                        preProcess = c(pp.list[i]),
                        tuneLength = 5,
                        trace = FALSE)
    
    ## predictions
    pred.train <- predict(model[[k]],train)
    pred.test  <- predict(model[[k]],test)
    Pred[,k]   <- c(pred.train,pred.test)
    
    ## Metrics performance
    SMAPE.train <- smape(pred.train, y_train)
    RRMSE.train <- RMSE(pred.train, y_train)/mean(pred.train)
    R2.train    <- cor(pred.train, y_train)^2
    
    SMAPE.test <- smape(pred.test, y_test)
    RRMSE.test <- RMSE(pred.test, y_test)/mean(pred.test)
    R2.test    <- cor(pred.test, y_test)^2
    
    metrics.train[k,] <- c(k, SMAPE.train, RRMSE.train, R2.train)
    metrics.test[k,] <- c(k, SMAPE.test, RRMSE.test, R2.test)
    
    k <- k + 1
    
    cat('PP:', pp.list[i], 'Model:', model.list[j], 'SMAPE:', SMAPE.test,
        'RRMSE:', RRMSE.test, 'R2:', R2.test, '\n')
  }
  save.image('base-results.RData')
}

metrics.test

xtable::xtable(metrics.test, digits = 4)


for (i in 1:6) {
  PO(data$y, Pred[,i])
}


# Stacking Training and Predictions---------------------------------
# setwd(ResultsDir)
# load('base-results.RData')

set.seed(1234)

stack.database <- list()
stack.database.train <- NA
stack.database.test  <- NA

meta.list <- c('svmRadial','cubist')

preprocess.list <- c("pca","corr")

stack <- list()
stack.pred.train <- NA
stack.pred.test  <- NA
stack.pred <- matrix(ncol = length(meta.list)*length(preprocess.list)*2, 
                     nrow = n)
colnames(stack.pred) <- c('PCA-SVRR-PCA','PCA-SVRR-CORR',
                          'PCA-CUBIST-PCA','PCA-CUBIST-CORR',
                          'CORR-SVRR-PCA','CORR-SVRR-CORR',
                          'CORR-CUBIST-PCA','CORR-CUBIST-CORR')

stack.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list)*2,
                              ncol = 4)
stack.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list)*2,
                              ncol = 4)
colnames(stack.Metrics) <- c("k","SMAPE","RRMSE","R2")
rownames(stack.Metrics) <- c('PCA-SVRR-PCA','PCA-SVRR-CORR',
                             'PCA-CUBIST-PCA','PCA-CUBIST-CORR',
                             'CORR-SVRR-PCA','CORR-SVRR-CORR',
                             'CORR-CUBIST-PCA','CORR-CUBIST-CORR')
f <- 1
k <- 1

for (m in 1:2) {
  stack.database[[m]] <- data.frame(data$y,Pred[,f:(f+3)])
  colnames(stack.database[[m]]) <- c("y","x1","x2","x3","x4")
  
  n <- dim(stack.database[[m]])[1]
  cut <- 0.7 * n
  
  stack.database.train <- stack.database[[m]][1:cut,]
  stack.database.test  <- tail(stack.database[[m]],n-cut)
  
  for (i in 1:length(meta.list)) {
    for (j in 1:length(preprocess.list)) {
      stack[[k]] <- train(y~.,data = stack.database.train,
                          method = meta.list[i],
                          trControl = control,
                          preProcess = preprocess.list[j],
                          tuneLength = 5,
                          trace = FALSE)
      
      
      stack.pred.train <- predict(stack[[k]],stack.database.train)
      stack.pred.test  <- predict(stack[[k]],stack.database.test)
      stack.pred[,k]   <- c(stack.pred.train,stack.pred.test)
      
      
      stack.SMAPE <- smape(stack.pred.test, y_test)
      stack.RRMSE <- RMSE(stack.pred.test, y_test)/mean(stack.pred.test)
      stack.R2    <- cor(stack.pred.test, y_test)^2
      
      stack.SMAPE.train <- smape(stack.pred.train, y_train)
      stack.RRMSE.train <- RMSE(stack.pred.train, y_train)/mean(stack.pred.train)
      stack.R2.train    <- cor(stack.pred.train, y_train)^2
      
      stack.Metrics.train[k,] <- c(k, 
                                   stack.SMAPE.train,
                                   stack.RRMSE.train,
                                   stack.R2.train)
      stack.Metrics[k,] <- c(k,
                             stack.SMAPE,
                             stack.RRMSE,
                             stack.R2)
      
      k <- k + 1
      
      cat("Model:", meta.list[i], "pp:", preprocess.list[j], 
          "SMAPE:", stack.SMAPE, "RRMSE:", stack.RRMSE,"R2:", stack.R2, "\n\n")
    }
  }
  save.image("stack-results.RData")
  f <- f + 4
}

stack.Metrics

xtable::xtable(stack.Metrics, digits = 4)

# Diebold-Mariano tests---------------------------------------------

PREDS1 <- matrix(ncol = 16, nrow = n-cut)

PREDS1[,1] <- tail(stack.pred[,1],n-cut)
PREDS1[,2] <- tail(stack.pred[,3],n-cut)
PREDS1[,3] <- tail(stack.pred[,2],n-cut)
PREDS1[,4] <- tail(stack.pred[,4],n-cut)
PREDS1[,5] <- tail(stack.pred[,5],n-cut)
PREDS1[,6] <- tail(stack.pred[,7],n-cut)
PREDS1[,7] <- tail(stack.pred[,6],n-cut)
PREDS1[,8] <- tail(stack.pred[,8],n-cut)

for (i in seq(dim(Pred)[2])) {
  PREDS1[,i+8] <- tail(Pred[,i],n-cut)
}

## 1 step
{
  h <- 1
  
  e <- matrix(ncol = 16, nrow = length(y_test))
  colnames(e) <- c('(A) SVRR-PCA-PCA',
                   '(B) CUBIST-PCA-PCA',
                   '(C) SVRR-PCA-CORR',
                   '(D) CUBIST-PCA-CORR',
                   '(E) SVRR-CORR-PCA',
                   '(F) CUBIST-CORR-PCA',
                   '(G) SVRR-CORR-CORR',
                   '(H) CUBIST-CORR-CORR',
                   '(I) pca-svr',
                   '(J) pca-mars',
                   '(K) pca-brnn',
                   '(L) pca-lm',
                   '(M) corr-svr',
                   '(N) corr-mars',
                   '(O) corr-brnn',
                   '(P) corr-lm')
  
  for (i in 1:dim(e)[2]) {
    e[,i] <- (y_test - PREDS1[,i])^2
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  colnames(DM.tvalue) <- c('(A)','(B)','(C)','(D)','(E)','(F)',
                           '(G)','(H)','(I)','(J)','(K)','(L)',
                           '(M)','(N)','(O)','(P)')
  rownames(DM.tvalue) <- c('(A) SVRR-PCA-PCA',
                           '(B) CUBIST-PCA-PCA',
                           '(C) SVRR-PCA-CORR',
                           '(D) CUBIST-PCA-CORR',
                           '(E) SVRR-CORR-PCA',
                           '(F) CUBIST-CORR-PCA',
                           '(G) SVRR-CORR-CORR',
                           '(H) CUBIST-CORR-CORR',
                           '(I) pca-svr',
                           '(J) pca-mars',
                           '(K) pca-brnn',
                           '(L) pca-lm',
                           '(M) corr-svr',
                           '(N) corr-mars',
                           '(O) corr-brnn',
                           '(P) corr-lm')
  colnames(DM.pvalue) <- c('(A)','(B)','(C)','(D)','(E)','(F)',
                           '(G)','(H)','(I)','(J)','(K)','(L)',
                           '(M)','(N)','(O)','(P)')
  rownames(DM.pvalue) <- c('(A) SVRR-PCA-PCA',
                           '(B) CUBIST-PCA-PCA',
                           '(C) SVRR-PCA-CORR',
                           '(D) CUBIST-PCA-CORR',
                           '(E) SVRR-CORR-PCA',
                           '(F) CUBIST-CORR-PCA',
                           '(G) SVRR-CORR-CORR',
                           '(H) CUBIST-CORR-CORR',
                           '(I) pca-svr',
                           '(J) pca-mars',
                           '(K) pca-brnn',
                           '(L) pca-lm',
                           '(M) corr-svr',
                           '(N) corr-mars',
                           '(O) corr-brnn',
                           '(P) corr-lm')
  
  for (i in 1:dim(e)[2]) {
    for (j in 1:dim(e)[2]) {
      if(i>=j) {
        DM.tvalue[i,j] <- NA
        DM.pvalue[i,j] <- NA
      }
      else {
        DMtest <- dm.test(e[,i],e[,j], h = h, power = 2)
        DM.tvalue[i,j] <- DMtest$statistic
        DM.pvalue[i,j] <- DMtest$p.value
      }
    }
  }
}


besttune <- list()
for (i in seq(stack)) {
  print(stack[[i]]$bestTune)
}
