# Starting Code ----------------------------------------------------
## Clear memory
rm(list=ls(all=TRUE))
Sys.setenv("LANGUAGE"="En")
Sys.setlocale("LC_ALL", "English")

## Set working directory
setwd("~/Doc/WCCI/")

BaseDir       <- getwd()
ResultsDir    <- paste(BaseDir, "Results-VMD", sep="/")

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
library(hht)
library(foreach)
library(iterators)
library(doParallel)
library(faraway)
library(vmd)
library(Quandl)

source("Plot2.R")
source('Plot_Recorte.R')
source('plot_IMF.R')
source('boxplot_error.R')

## Cores Cluster
ncl <- detectCores();ncl
cl  <- makeCluster(ncl-1);registerDoParallel(cl)
# stopImplicitCluster() # Stop

# Data treatment ---------------------------------------------------

## Load data

url <- "https://raw.githubusercontent.com/ByronKKing/Time-Series-R/master/bitcoin-2016-02-21.csv"

dataset <- read.csv(url, sep = ",")

## Decomposition

vmd.results <- vmd(signal = dataset$Close,
                   tol = 1e-6,
                   DC = FALSE,
                   K = 5)

v <- vmd.results$as.data.frame()

## PACF and autoarima

PACF <- list()
autoarima <- list()

for (i in 1:vmd.results$K) {
  PACF[[i]] <- pacf(v[,i+2], main = paste("IMF",i))
  autoarima[[i]] <- auto.arima(v[,i+2])
  print(autoarima[[i]])
}


IMF <- matrix(ncol = vmd.results$K, nrow = dim(dataset)[1])
for (i in 1:vmd.results$K) {
  IMF[,i] <- v[,i+2]
}

colnames(IMF) <- c("IMF1","IMF2","IMF3","IMF4","IMF5")

## dataframes

lag <- 1

IMF <- list()

for (i in 1:vmd.results$K) {
  IMF[[i]] <- data.frame(v[,i+2][(lag+1):(dim(v)[1])],
                         dataset[,2][lag:(dim(v)[1]-lag)],
                         dataset[,3][lag:(dim(v)[1]-lag)],
                         dataset[,4][lag:(dim(v)[1]-lag)])
  colnames(IMF[[i]]) <- c('y','Open','High','Low')
}

## Training and Test sets

IMF.train  <- list()
IMF.test   <- list()
IMF.xtrain <- list()
IMF.ytrain <- list()
IMF.xtest  <- list()
IMF.ytest  <- list()

for (i in 1:length(IMF)) {
  n <- dim(IMF[[i]])[1]
  cut <- 0.7*n
  
  IMF.train[[i]] <- IMF[[i]][1:cut,]
  IMF.test[[i]]  <- tail(IMF[[i]],n-cut)
  
  IMF.xtrain[[i]] <- IMF.train[[i]][,-1]
  IMF.ytrain[[i]] <- IMF.train[[i]][,1]
  
  IMF.xtest[[i]] <- IMF.test[[i]][,-1]
  IMF.ytest[[i]] <- IMF.test[[i]][,1]
}

res <- list()
VIF <- list()
for (i in 1:length(IMF)) {
  res[[i]] <- cor(IMF[[i]])
  cat("\nIMF",i, '\n',sep = '')
  print(round(res[[i]],4))
  
  VIF[[i]] <- vif(IMF[[i]])
  cat("\nVIF",i, '\n',sep = '')
  print(round(VIF[[i]],4))
}

setwd(ResultsDir)
save.image("VMD-data.RData")

# VMD Training and Predictions -----------------------------------
# setwd(ResultsDir)
# load("VMD-data.RData")

set.seed(1234)

control <- trainControl(method = "timeslice",
                        initialWindow = 0.8*dim(IMF.train[[1]])[1],
                        horizon = 0.1*dim(IMF.train[[1]])[1],
                        fixedWindow = FALSE,
                        allowParallel = TRUE,
                        savePredictions = 'final',
                        verboseIter = FALSE)

model.list <- c('knn', 
                'svmLinear2', 
                'nnet',
                'glm') 

## Define objects 
{
  IMF.model      <- list()
  pred.IMF.train <- NA
  pred.IMF.test  <- NA
  pred.IMF       <- list()
  k <- 1
}

for (i in 1:length(model.list)) {
  for (j in 1:length(IMF)) {
    IMF.model[[k]] <- train(y~., data = IMF.train[[j]],
                            method = model.list[i],
                            trControl = control,
                            preProcess = c("BoxCox"),
                            # tuneLenght = 5,
                            trace = FALSE)
    
    ### Prediction
    
    pred.IMF.train   <- predict(IMF.model[[k]],IMF.train[[j]])
    pred.IMF.test    <- predict(IMF.model[[k]],IMF.test[[j]])
    pred.IMF[[k]]    <- data.frame(c(pred.IMF.train,pred.IMF.test))
    
    cat("\nModel: ", model.list[i], "\tIMF", j, "\t",
        as.character(Sys.time()), sep = '')
    
    k <- k + 1
  }
}

save.image("VMD-training.RData")

# count <- c(1:length(model.list))

# combs <- expand.grid(count,count,count,count,count)

combs <- rbind(c(1,1,1,1,1),
               c(2,2,2,2,2),
               c(3,3,3,3,3),
               c(4,4,4,4,4))

colnames(combs) <- c("IMF1","IMF2","IMF3","IMF4","IMF5")

### Creating Obs.train and Obs.test

Obs   <- dataset$Close[(lag+1):(dim(dataset)[1])]
n <- dim(IMF[[1]])[1]
cut <- 0.7 * n
Obs.train <- Obs[1:cut]
Obs.test  <- tail(Obs,n-cut)


VMD.Metrics <- matrix(nrow = dim(combs)[1],ncol = 4)
VMD.Metrics.train <- matrix(nrow = dim(combs)[1],ncol = 4)
colnames(VMD.Metrics)       <- c("i","SMAPE","RRMSE","R2")
colnames(VMD.Metrics.train) <- c("i","SMAPE","RRMSE","R2")
rownames(VMD.Metrics)       <- model.list
rownames(VMD.Metrics.train) <- model.list

VMD.Prediction <- matrix(0,nrow = n, ncol = dim(combs)[1])

k <- 1

for (i in 1:length(model.list)) {
  for (j in 1:dim(combs)[2]) {
    VMD.Prediction[,i] <- as.vector(VMD.Prediction[,i]) + 
      as.vector(pred.IMF[[k]][,1])
    k <- k + 1
  }
}

for (i in 1:dim(combs)[1]) {
  for (j in 1:dim(VMD.Prediction)[1]) {
    if (VMD.Prediction[j,i] < 0) {
      VMD.Prediction[j,i] <- 0
    }
  }
  
  VMD.Prediction.train <- VMD.Prediction[,i][1:cut]
  VMD.Prediction.test  <- tail(VMD.Prediction[,i],n-cut)
  
  # #Metrics
  VMD.SMAPE <- smape(VMD.Prediction.test, Obs.test)
  VMD.RRMSE <- RMSE(VMD.Prediction.test, Obs.test)/mean(VMD.Prediction.test)
  VMD.R2    <- cor(VMD.Prediction.test, Obs.test)^2
  
  VMD.SMAPE.train <- smape(VMD.Prediction.train, Obs.train)
  VMD.RRMSE.train <- RMSE(VMD.Prediction.train, Obs.train)/mean(VMD.Prediction.train)
  VMD.R2.train    <- cor(VMD.Prediction.train, Obs.train)^2
  
  VMD.Metrics.train[i,] <- c(i, 
                             VMD.SMAPE.train,
                             VMD.RRMSE.train,
                             VMD.R2.train)
  VMD.Metrics[i,] <- c(i,
                       VMD.SMAPE,
                       VMD.RRMSE,
                       VMD.R2)
}

save.image("Results-VMD.RData")

VMD.Metrics
xtable::xtable(VMD.Metrics, digits = 4)

# Stacking Training and Predictions---------------------------------
# setwd(ResultsDir)
# load("Results-VMD.RData")
set.seed(1234)
stack.database <- data.frame(Obs,VMD.Prediction)
colnames(stack.database) <- c("y","x1","x2","x3","x4")

n <- dim(stack.database)[1]
cut <- 0.7 * n

stack.database.train <- stack.database[1:cut,]
stack.database.test  <- tail(stack.database,n-cut)

meta.list <- c('svmRadial','brnn','cubist')

preprocess.list <- c("corr","pca","BoxCox")

stack <- list()
stack.pred.train <- NA
stack.pred.test  <- NA
stack.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                     nrow = n)

stack.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                              ncol = 4)
stack.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                              ncol = 4)
colnames(stack.Metrics) <- c("k","SMAPE","RRMSE","R2")
rownames(stack.Metrics) <- c("svr+corr","svr+pca","svr+BoxCox",
                             "brnn+corr","brnn+pca","brnn+BoxCox",
                             "cubist+corr","cubist+pca","cubist+BoxCox")

k <- 1

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
    
    
    stack.SMAPE <- smape(stack.pred.test, Obs.test)
    stack.RRMSE <- RMSE(stack.pred.test, Obs.test)/mean(stack.pred.test)
    stack.R2    <- cor(stack.pred.test, Obs.test)^2
    
    stack.SMAPE.train <- smape(stack.pred.train, Obs.train)
    stack.RRMSE.train <- RMSE(stack.pred.train, Obs.train)/mean(stack.pred.train)
    stack.R2.train    <- cor(stack.pred.train, Obs.train)^2
    
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
  save.image("stack-VMD-results.RData")
}

stack.Metrics

xtable::xtable(stack.Metrics, digits = 4)

# VMD step-ahead Predictions--------------------------------------
setwd(ResultsDir)
load("Results-VMD.RData")

## 2 steps
{
### Creating dataframes com lags e inputs
{
  lag <- 2
  
  IMF.2step        <- list()
  IMF.train.2step  <- list()
  IMF.test.2step   <- list()
  IMF.xtrain.2step <- list()
  IMF.ytrain.2step <- list()
  IMF.xtest.2step  <- list()
  IMF.ytest.2step  <- list()
} 

for (i in seq(vmd.results$K)) {
  IMF.2step[[i]] <- data.frame(v[,i+2][(lag+1):(dim(v)[1])],
                               v[,i+2][(lag-0):(dim(v)[1]-lag+1)],
                               v[,i+2][(lag-1):(dim(v)[1]-lag+0)],
                               dataset[,2][(lag-1):(dim(v)[1]-lag+0)],
                               dataset[,3][(lag-1):(dim(v)[1]-lag+0)],
                               dataset[,4][(lag-1):(dim(v)[1]-lag+0)])
  colnames(IMF.2step[[i]]) <- c('y(t)','y(t-1)','y(t-2)','Open','High','Low')
  
  n <- dim(IMF.2step[[i]])[1]
  cut <- 0.7*n
  
  IMF.train.2step[[i]] <- IMF.2step[[i]][1:cut,]
  IMF.test.2step[[i]]  <- tail(IMF.2step[[i]],n-cut)
  
  IMF.xtrain.2step[[i]] <- IMF.train.2step[[i]][,-1]
  IMF.ytrain.2step[[i]] <- IMF.train.2step[[i]][,1]
  
  IMF.xtest.2step[[i]] <- IMF.test.2step[[i]][,-1]
  IMF.ytest.2step[[i]] <- IMF.test.2step[[i]][,1]
}

{
  IMF1.model <- list(IMF.model[[1]],
                     IMF.model[[6]],
                     IMF.model[[11]],
                     IMF.model[[16]])
  
  IMF2.model <- list(IMF.model[[2]],
                     IMF.model[[7]],
                     IMF.model[[12]],
                     IMF.model[[17]])
  
  IMF3.model <- list(IMF.model[[3]],
                     IMF.model[[8]],
                     IMF.model[[13]],
                     IMF.model[[18]])
  
  IMF4.model <- list(IMF.model[[4]],
                     IMF.model[[9]],
                     IMF.model[[14]],
                     IMF.model[[19]])
  
  IMF5.model <- list(IMF.model[[5]],
                     IMF.model[[10]],
                     IMF.model[[15]],
                     IMF.model[[20]])
  
  Obs.train <- dataset[1:cut,5]
  Obs.test  <- tail(dataset[,5],n-cut)
  Obs <- c(Obs.train,Obs.test)
}

### Recursive prediction

{
  h <- 2
  PTRmo <- list()
  PTEmo <- list()
}

for (m in 1:length(model.list)) {
  IMF1.xtrainm <- as.data.frame(IMF.xtrain.2step[[1]])
  IMF2.xtrainm <- as.data.frame(IMF.xtrain.2step[[2]])
  IMF3.xtrainm <- as.data.frame(IMF.xtrain.2step[[3]])
  IMF4.xtrainm <- as.data.frame(IMF.xtrain.2step[[4]])
  IMF5.xtrainm <- as.data.frame(IMF.xtrain.2step[[5]])
  
  IMF1.xtestm <- as.data.frame(IMF.xtest.2step[[1]])
  IMF2.xtestm <- as.data.frame(IMF.xtest.2step[[2]])
  IMF3.xtestm <- as.data.frame(IMF.xtest.2step[[3]])
  IMF4.xtestm <- as.data.frame(IMF.xtest.2step[[4]])
  IMF5.xtestm <- as.data.frame(IMF.xtest.2step[[5]])
  
  PTRmo[[m]] <- matrix(ncol = 5, nrow = dim(IMF.train.2step[[1]])[1])
  PTEmo[[m]] <- matrix(ncol = 5, nrow = dim(IMF.test.2step[[1]])[1])
  
  ### Train
  for(p in 1:dim(IMF.train.2step[[1]])[1]){
    if(p%%h !=0){
      PTRmo[[m]][p,1] <- predict(IMF1.model[[m]],IMF1.xtrainm[p,])
      IMF1.xtrainm[p+1,1] <- PTRmo[[m]][p,1]
      
      PTRmo[[m]][p,2] <- predict(IMF2.model[[m]],IMF2.xtrainm[p,])
      IMF2.xtrainm[p+1,1] <- PTRmo[[m]][p,2]
      
      PTRmo[[m]][p,3] <- predict(IMF3.model[[m]],IMF3.xtrainm[p,])
      IMF3.xtrainm[p+1,1] <- PTRmo[[m]][p,3]
      
      PTRmo[[m]][p,4] <- predict(IMF4.model[[m]],IMF4.xtrainm[p,])
      IMF4.xtrainm[p+1,1] <- PTRmo[[m]][p,4]
      
      PTRmo[[m]][p,5] <- predict(IMF5.model[[m]],IMF5.xtrainm[p,])
      IMF5.xtrainm[p+1,1] <- PTRmo[[m]][p,5]
    }
    else{
      PTRmo[[m]][p,1] <- predict(IMF1.model[[m]],IMF1.xtrainm[p,])
      PTRmo[[m]][p,2] <- predict(IMF2.model[[m]],IMF2.xtrainm[p,])
      PTRmo[[m]][p,3] <- predict(IMF3.model[[m]],IMF3.xtrainm[p,])
      PTRmo[[m]][p,4] <- predict(IMF4.model[[m]],IMF4.xtrainm[p,])
      PTRmo[[m]][p,5] <- predict(IMF5.model[[m]],IMF5.xtrainm[p,])
    }
  }
  
  ### Test
  for(p in 1:dim(IMF.test.2step[[1]])[1]){
    if(p%%h !=0){
      PTEmo[[m]][p,1] <- predict(IMF1.model[[m]],IMF1.xtestm[p,])
      IMF1.xtestm[p+1,1] <- PTEmo[[m]][p,1]
      
      PTEmo[[m]][p,2] <- predict(IMF2.model[[m]],IMF2.xtestm[p,])
      IMF2.xtestm[p+1,1] <- PTEmo[[m]][p,2]
      
      PTEmo[[m]][p,3] <- predict(IMF3.model[[m]],IMF3.xtestm[p,])
      IMF3.xtestm[p+1,1] <- PTEmo[[m]][p,3]
      
      PTEmo[[m]][p,4] <- predict(IMF4.model[[m]],IMF4.xtestm[p,])
      IMF4.xtestm[p+1,1] <- PTEmo[[m]][p,4]
      
      PTEmo[[m]][p,5] <- predict(IMF5.model[[m]],IMF5.xtestm[p,])
      IMF5.xtestm[p+1,1] <- PTEmo[[m]][p,5]
    }
    else{
      PTEmo[[m]][p,1] <- predict(IMF1.model[[m]],IMF1.xtestm[p,])
      PTEmo[[m]][p,2] <- predict(IMF2.model[[m]],IMF2.xtestm[p,])
      PTEmo[[m]][p,3] <- predict(IMF3.model[[m]],IMF3.xtestm[p,])
      PTEmo[[m]][p,4] <- predict(IMF4.model[[m]],IMF4.xtestm[p,])
      PTEmo[[m]][p,5] <- predict(IMF5.model[[m]],IMF5.xtestm[p,])
    }
  }
  
  cat("Model:", model.list[m], m/length(model.list)*100,"%\n")
}

{
  Metrics2.train <- matrix(nrow = dim(combs)[1],ncol = 4)
  Metrics2       <- matrix(nrow = dim(combs)[1],ncol = 4)
  colnames(Metrics2) <- c("comb","SMAPE","RRMSE","R2")
  rownames(Metrics2) <- model.list
  
  Pred2step.train <- matrix(nrow = dim(IMF.train.2step[[1]])[1], ncol = dim(combs)[1])
  Pred2step.test  <- matrix(nrow = dim(IMF.test.2step[[1]])[1], ncol = dim(combs)[1])
  Pred2step <- matrix(nrow = n, ncol = dim(combs)[1])
}

for (c in 1:dim(combs)[1]) {
  ### Recomposing the prediction
  Pred2step.train[,c] <- (PTRmo[[combs[c,1]]][,1] + 
                          PTRmo[[combs[c,2]]][,2] + 
                          PTRmo[[combs[c,3]]][,3] +
                          PTRmo[[combs[c,4]]][,4] +
                          PTRmo[[combs[c,5]]][,5])
  
  Pred2step.test[,c] <- (PTEmo[[combs[c,1]]][,1] + 
                         PTEmo[[combs[c,2]]][,2] + 
                         PTEmo[[combs[c,3]]][,3] +
                         PTEmo[[combs[c,4]]][,4] +
                         PTEmo[[combs[c,5]]][,5])
  
  ### Avoiding negative values
  for (j in 1:dim(Pred2step.train)[1]) {
    if (Pred2step.train[j,c] < 0) {
      Pred2step.train[j,c] <- 0
    }
  }
  for (j in 1:dim(Pred2step.test)[1]) {
    if (Pred2step.test[j,c] < 0) {
      Pred2step.test[j,c] <- 0
    }
  }
  
  ### Predictions
  Pred2step[,c] <- c(Pred2step.train[,c],Pred2step.test[,c])
  
  pred2.SMAPE <- smape(Pred2step.test[,c], Obs.test)
  pred2.RRMSE <- RMSE(Pred2step.test[,c], Obs.test)/mean(Pred2step.test[,c])
  pred2.R2    <- cor(Pred2step.test[,c], Obs.test)^2
  
  pred2.SMAPE.train <- smape(Pred2step.train[,c], Obs.train)
  pred2.RRMSE.train <- RMSE(Pred2step.train[,c], Obs.train)/mean(Pred2step.train[,c])
  pred2.R2.train    <- cor(Pred2step.train[,c], Obs.train)^2
  
  Metrics2.train[c,] <- cbind(c,pred2.SMAPE.train,pred2.RRMSE.train,pred2.R2.train)
  Metrics2[c,] <- cbind(c,pred2.SMAPE,pred2.RRMSE,pred2.R2)
}

xtable::xtable(Metrics2, digits = 4)

# save.image("2step-final.RData")
}

## 3 steps
{
  ### Creating dataframes com lags e inputs
  {
    lag <- 3
    
    IMF.3step        <- list()
    IMF.train.3step  <- list()
    IMF.test.3step   <- list()
    IMF.xtrain.3step <- list()
    IMF.ytrain.3step <- list()
    IMF.xtest.3step  <- list()
    IMF.ytest.3step  <- list()
  } 
  
  for (i in seq(vmd.results$K)) {
    IMF.3step[[i]] <- data.frame(v[,i+2][(lag+1):(dim(v)[1])],
                                 v[,i+2][(lag-0):(dim(v)[1]-lag+2)],
                                 v[,i+2][(lag-1):(dim(v)[1]-lag+1)],
                                 v[,i+2][(lag-2):(dim(v)[1]-lag+0)],
                                 dataset[,2][(lag-2):(dim(v)[1]-lag+0)],
                                 dataset[,3][(lag-2):(dim(v)[1]-lag+0)],
                                 dataset[,4][(lag-2):(dim(v)[1]-lag+0)])
    colnames(IMF.3step[[i]]) <- c('y(t)','y(t-1)','y(t-2)',
                                  'y(t-3)',
                                  'Open','High','Low')
    
    n <- dim(IMF.3step[[i]])[1]
    cut <- 0.7*n
    
    IMF.train.3step[[i]] <- IMF.3step[[i]][1:cut,]
    IMF.test.3step[[i]]  <- tail(IMF.3step[[i]],n-cut)
    
    IMF.xtrain.3step[[i]] <- IMF.train.3step[[i]][,-1]
    IMF.ytrain.3step[[i]] <- IMF.train.3step[[i]][,1]
    
    IMF.xtest.3step[[i]] <- IMF.test.3step[[i]][,-1]
    IMF.ytest.3step[[i]] <- IMF.test.3step[[i]][,1]
  }
  
  {
    IMF1.model <- list(IMF.model[[1]],
                       IMF.model[[6]],
                       IMF.model[[11]],
                       IMF.model[[16]])
    
    IMF2.model <- list(IMF.model[[2]],
                       IMF.model[[7]],
                       IMF.model[[12]],
                       IMF.model[[17]])
    
    IMF3.model <- list(IMF.model[[3]],
                       IMF.model[[8]],
                       IMF.model[[13]],
                       IMF.model[[18]])
    
    IMF4.model <- list(IMF.model[[4]],
                       IMF.model[[9]],
                       IMF.model[[14]],
                       IMF.model[[19]])
    
    IMF5.model <- list(IMF.model[[5]],
                       IMF.model[[10]],
                       IMF.model[[15]],
                       IMF.model[[20]])
    
    Obs.train <- dataset[1:cut,5]
    Obs.test  <- tail(dataset[,5],n-cut)
    Obs <- c(Obs.train,Obs.test)
  }
  
  ### Recursive prediction
  
  {
    h <- 3
    obsTR <- list()
    obsTE <- list()
    PTRmo <- list()
    PTEmo <- list()
    model <- list()
    Comp.train <- list()
    Comp.test <- list()
    k <- 1
  }
  
  for (m in 1:length(model.list)) {
    
    IMF1.trainm <- as.data.frame(IMF.xtrain.3step[[1]])
    IMF2.trainm <- as.data.frame(IMF.xtrain.3step[[2]])
    IMF3.trainm <- as.data.frame(IMF.xtrain.3step[[3]])
    IMF4.trainm <- as.data.frame(IMF.xtrain.3step[[4]])
    IMF5.trainm <- as.data.frame(IMF.xtrain.3step[[5]])
    
    Comp.train[[m]] <- list(IMF1.trainm,IMF2.trainm,IMF3.trainm,IMF4.trainm,IMF5.trainm)
    
    IMF1.testm <- as.data.frame(IMF.xtest.3step[[1]])
    IMF2.testm <- as.data.frame(IMF.xtest.3step[[2]])
    IMF3.testm <- as.data.frame(IMF.xtest.3step[[3]])
    IMF4.testm <- as.data.frame(IMF.xtest.3step[[4]])
    IMF5.testm <- as.data.frame(IMF.xtest.3step[[5]])
    
    Comp.test[[m]] <- list(IMF1.testm,IMF2.testm,IMF3.testm,IMF4.testm,IMF5.testm)
    
    model[[m]] <- list(IMF1.model[[m]],IMF2.model[[m]],IMF3.model[[m]],
                       IMF4.model[[m]],IMF5.model[[m]])
    
    PTRmo[[m]] <- matrix(ncol = length(IMF), nrow = dim(IMF1.trainm)[1])
    PTEmo[[m]] <- matrix(ncol = length(IMF), nrow = dim(IMF1.testm)[1])
    
    for (c in 1:length(Comp.train[[m]])) {
      obsTR[[c]] <- matrix(seq(1,dim(Comp.train[[m]][[c]])[1]+2,1),ncol = h, byrow = TRUE)
      obsTE[[c]] <- matrix(seq(1,dim(Comp.test[[m]][[c]])[1]+2,1),ncol = h, byrow = TRUE)
      
      # Train
      for (N in 1:h) {
        for (v in 1:dim(obsTR[[c]])[1]) {
          p <- obsTR[[c]][v,N]
          if (p <= dim(Comp.train[[m]][[c]])[1]) {
            if(p==obsTR[[c]][v,1]){
              PTRmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
            else if(p==obsTR[[c]][v,2]) {
              Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
              PTRmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
            else {
              Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
              Comp.train[[m]][[c]][p,2] <- Comp.train[[m]][[c]][p-1,1]
              
              PTRmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
          }
          else {
            break
          }
        }
      }
      
      # Test
      for (N in 1:h) {
        for (v in 1:dim(obsTE[[c]])[1]) {
          p <- obsTE[[c]][v,N]
          if (p <= dim(Comp.test[[m]][[c]])[1]) {
            if(p==obsTE[[c]][v,1]){
              PTEmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
            else if(p==obsTE[[c]][v,2]) {
              Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
              PTEmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
            else {
              Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
              Comp.test[[m]][[c]][p,2] <- Comp.test[[m]][[c]][p-1,1]
              
              PTEmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
          }
          else {
            break
          }
        }
      }
      
      cat("Model: ", model.list[m], "\tComp: ", c , "\t", 
          (k/(length(model.list)*length(Comp.train[[m]])))*100,"%\n", sep = "")
      
      k <- k + 1
    }
  }
  
  {
    VMD3.Metrics <- matrix(nrow = dim(combs)[1],ncol = 4)
    VMD3.Metrics.train <- matrix(nrow = dim(combs)[1],ncol = 4)
    colnames(VMD3.Metrics)       <- c("i","SMAPE","RRMSE","R2")
    colnames(VMD3.Metrics.train) <- c("i","SMAPE","RRMSE","R2")
    rownames(VMD3.Metrics)       <- model.list
    rownames(VMD3.Metrics.train) <- model.list
    
    
    VMD3.Pred.train <- matrix(nrow = dim(IMF.xtrain.3step[[1]])[1], ncol = dim(combs)[1])
    VMD3.Pred.test  <- matrix(nrow = dim(IMF.xtest.3step[[1]])[1], ncol = dim(combs)[1])
    VMD3.Prediction <- matrix(nrow = n, ncol = dim(combs)[1])
  }
  
  for (i in 1:dim(combs)[1]) {
    VMD3.Pred.train[,i] <- (PTRmo[[combs[i,1]]][,1] + 
                              PTRmo[[combs[i,2]]][,2] + 
                              PTRmo[[combs[i,3]]][,3] +
                              PTRmo[[combs[i,4]]][,4] + 
                              PTRmo[[combs[i,5]]][,5])
    
    VMD3.Pred.test[,i] <- (PTEmo[[combs[i,1]]][,1] + 
                             PTEmo[[combs[i,2]]][,2] + 
                             PTEmo[[combs[i,3]]][,3] +
                             PTEmo[[combs[i,4]]][,4] + 
                             PTEmo[[combs[i,5]]][,5])
    
    ### Avoiding negative values
    for (j in 1:dim(VMD3.Pred.train)[1]) {
      if (VMD3.Pred.train[j,i] < 0) {
        VMD3.Pred.train[j,i] <- 0
      }
    }
    for (j in 1:dim(VMD3.Pred.test)[1]) {
      if (VMD3.Pred.test[j,i] < 0) {
        VMD3.Pred.test[j,i] <- 0
      }
    }
    
    VMD3.Prediction[,i] <- c(VMD3.Pred.train[,i],VMD3.Pred.test[,i])
    
    # #Metrics
    VMD3.SMAPE <- smape(VMD3.Pred.test[,i], Obs.test)
    VMD3.RRMSE <- RMSE(VMD3.Pred.test[,i], Obs.test)/mean(VMD3.Pred.test[,i])
    VMD3.R2    <- cor(VMD3.Pred.test[,i], Obs.test)^2
    
    VMD3.SMAPE.train <- smape(VMD3.Pred.train[,i], Obs.train)
    VMD3.RRMSE.train <- RMSE(VMD3.Pred.train[,i], Obs.train)/mean(VMD3.Pred.train[,i])
    VMD3.R2.train    <- cor(VMD3.Pred.train[,i], Obs.train)^2
    
    VMD3.Metrics.train[i,] <- c(i, 
                                VMD3.SMAPE.train,
                                VMD3.RRMSE.train,
                                VMD3.R2.train)
    VMD3.Metrics[i,] <- c(i,
                          VMD3.SMAPE,
                          VMD3.RRMSE,
                          VMD3.R2)
  }
  
  xtable::xtable(VMD3.Metrics, digits = 4)
  
  save.image("3step-final.RData")
}

## 4 steps
{
### Creating dataframes com lags e inputs
{
  lag <- 4
  
  IMF.4step        <- list()
  IMF.train.4step  <- list()
  IMF.test.4step   <- list()
  IMF.xtrain.4step <- list()
  IMF.ytrain.4step <- list()
  IMF.xtest.4step  <- list()
  IMF.ytest.4step  <- list()
} 

for (i in seq(vmd.results$K)) {
  IMF.4step[[i]] <- data.frame(v[,i+2][(lag+1):(dim(v)[1])],
                               v[,i+2][(lag-0):(dim(v)[1]-lag+3)],
                               v[,i+2][(lag-1):(dim(v)[1]-lag+2)],
                               v[,i+2][(lag-2):(dim(v)[1]-lag+1)],
                               v[,i+2][(lag-3):(dim(v)[1]-lag+0)],
                               dataset[,2][(lag-3):(dim(v)[1]-lag+0)],
                               dataset[,3][(lag-3):(dim(v)[1]-lag+0)],
                               dataset[,4][(lag-3):(dim(v)[1]-lag+0)])
  colnames(IMF.4step[[i]]) <- c('y(t)','y(t-1)','y(t-2)',
                                'y(t-3)','y(t-4)',
                                'Open','High','Low')
  
  n <- dim(IMF.4step[[i]])[1]
  cut <- 0.7*n
  
  IMF.train.4step[[i]] <- IMF.4step[[i]][1:cut,]
  IMF.test.4step[[i]]  <- tail(IMF.4step[[i]],n-cut)
  
  IMF.xtrain.4step[[i]] <- IMF.train.4step[[i]][,-1]
  IMF.ytrain.4step[[i]] <- IMF.train.4step[[i]][,1]
  
  IMF.xtest.4step[[i]] <- IMF.test.4step[[i]][,-1]
  IMF.ytest.4step[[i]] <- IMF.test.4step[[i]][,1]
}

{
  IMF1.model <- list(IMF.model[[1]],
                     IMF.model[[6]],
                     IMF.model[[11]],
                     IMF.model[[16]])
  
  IMF2.model <- list(IMF.model[[2]],
                     IMF.model[[7]],
                     IMF.model[[12]],
                     IMF.model[[17]])
  
  IMF3.model <- list(IMF.model[[3]],
                     IMF.model[[8]],
                     IMF.model[[13]],
                     IMF.model[[18]])
  
  IMF4.model <- list(IMF.model[[4]],
                     IMF.model[[9]],
                     IMF.model[[14]],
                     IMF.model[[19]])
  
  IMF5.model <- list(IMF.model[[5]],
                     IMF.model[[10]],
                     IMF.model[[15]],
                     IMF.model[[20]])
  
  Obs.train <- dataset[1:cut,5]
  Obs.test  <- tail(dataset[,5],n-cut)
  Obs <- c(Obs.train,Obs.test)
}

### Recursive prediction

{
  h <- 4
  obsTR <- list()
  obsTE <- list()
  PTRmo <- list()
  PTEmo <- list()
  model <- list()
  Comp.train <- list()
  Comp.test <- list()
  k <- 1
}

for (m in 1:length(model.list)) {
  
  IMF1.trainm <- as.data.frame(IMF.xtrain.4step[[1]])
  IMF2.trainm <- as.data.frame(IMF.xtrain.4step[[2]])
  IMF3.trainm <- as.data.frame(IMF.xtrain.4step[[3]])
  IMF4.trainm <- as.data.frame(IMF.xtrain.4step[[4]])
  IMF5.trainm <- as.data.frame(IMF.xtrain.4step[[5]])
  
  Comp.train[[m]] <- list(IMF1.trainm,IMF2.trainm,IMF3.trainm,IMF4.trainm,IMF5.trainm)
  
  IMF1.testm <- as.data.frame(IMF.xtest.4step[[1]])
  IMF2.testm <- as.data.frame(IMF.xtest.4step[[2]])
  IMF3.testm <- as.data.frame(IMF.xtest.4step[[3]])
  IMF4.testm <- as.data.frame(IMF.xtest.4step[[4]])
  IMF5.testm <- as.data.frame(IMF.xtest.4step[[5]])
  
  Comp.test[[m]] <- list(IMF1.testm,IMF2.testm,IMF3.testm,IMF4.testm,IMF5.testm)
  
  model[[m]] <- list(IMF1.model[[m]],IMF2.model[[m]],IMF3.model[[m]],
                     IMF4.model[[m]],IMF5.model[[m]])
  
  PTRmo[[m]] <- matrix(ncol = length(IMF), nrow = dim(IMF1.trainm)[1])
  PTEmo[[m]] <- matrix(ncol = length(IMF), nrow = dim(IMF1.testm)[1])
  
  for (c in 1:length(Comp.train[[m]])) {
    obsTR[[c]] <- matrix(seq(1,dim(Comp.train[[m]][[c]])[1],1),ncol = h, byrow = TRUE)
    obsTE[[c]] <- matrix(seq(1,dim(Comp.test[[m]][[c]])[1]+3,1),ncol = h, byrow = TRUE)
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR[[c]])[1]) {
        p <- obsTR[[c]][v,N]
        if (p <= dim(Comp.train[[m]][[c]])[1]) {
          if(p==obsTR[[c]][v,1]){
            PTRmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
          }
          else if(p==obsTR[[c]][v,2]) {
            Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
            PTRmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
          }
          else if(p==obsTR[[c]][v,3]) {
            Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
            Comp.train[[m]][[c]][p,2] <- Comp.train[[m]][[c]][p-1,1]
            
            PTRmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
          }
          else {
            Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
            Comp.train[[m]][[c]][p,2] <- Comp.train[[m]][[c]][p-1,1]
            Comp.train[[m]][[c]][p,3] <- Comp.train[[m]][[c]][p-1,2]
            
            PTRmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
          }
        }
        else {
          break
        }
      }
    }
    
    # Test
    for (N in 1:h) {
      for (v in 1:dim(obsTE[[c]])[1]) {
        p <- obsTE[[c]][v,N]
        if (p <= dim(Comp.test[[m]][[c]])[1]) {
          if(p==obsTE[[c]][v,1]){
            PTEmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
          }
          else if(p==obsTE[[c]][v,2]) {
            Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
            PTEmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
          }
          else if(p==obsTE[[c]][v,3]) {
            Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
            Comp.test[[m]][[c]][p,2] <- Comp.test[[m]][[c]][p-1,1]
            
            PTEmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
          }
          else {
            Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
            Comp.test[[m]][[c]][p,2] <- Comp.test[[m]][[c]][p-1,1]
            Comp.test[[m]][[c]][p,3] <- Comp.test[[m]][[c]][p-1,2]
            
            PTEmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
          }
        }
        else {
          break
        }
      }
    }
    
    cat("Model: ", model.list[m], "\tComp: ", c , "\t", 
        (k/(length(model.list)*length(Comp.train[[m]])))*100,"%\n", sep = "")
    
    k <- k + 1
  }
}

{
  VMD4.Metrics <- matrix(nrow = dim(combs)[1],ncol = 4)
  VMD4.Metrics.train <- matrix(nrow = dim(combs)[1],ncol = 4)
  colnames(VMD4.Metrics)       <- c("i","SMAPE","RRMSE","R2")
  colnames(VMD4.Metrics.train) <- c("i","SMAPE","RRMSE","R2")
  rownames(VMD4.Metrics)       <- model.list
  rownames(VMD4.Metrics.train) <- model.list
  
  
  VMD4.Pred.train <- matrix(nrow = dim(IMF.xtrain.4step[[1]])[1], ncol = dim(combs)[1])
  VMD4.Pred.test  <- matrix(nrow = dim(IMF.xtest.4step[[1]])[1], ncol = dim(combs)[1])
  VMD4.Prediction <- matrix(nrow = n, ncol = dim(combs)[1])
}

for (i in 1:dim(combs)[1]) {
  VMD4.Pred.train[,i] <- (PTRmo[[combs[i,1]]][,1] + 
                              PTRmo[[combs[i,2]]][,2] + 
                              PTRmo[[combs[i,3]]][,3] +
                              PTRmo[[combs[i,4]]][,4] + 
                              PTRmo[[combs[i,5]]][,5])
  
  VMD4.Pred.test[,i] <- (PTEmo[[combs[i,1]]][,1] + 
                             PTEmo[[combs[i,2]]][,2] + 
                             PTEmo[[combs[i,3]]][,3] +
                             PTEmo[[combs[i,4]]][,4] + 
                             PTEmo[[combs[i,5]]][,5])
  
  ### Avoiding negative values
  for (j in 1:dim(VMD4.Pred.train)[1]) {
    if (VMD4.Pred.train[j,i] < 0) {
      VMD4.Pred.train[j,i] <- 0
    }
  }
  for (j in 1:dim(VMD4.Pred.test)[1]) {
    if (VMD4.Pred.test[j,i] < 0) {
      VMD4.Pred.test[j,i] <- 0
    }
  }
  
  VMD4.Prediction[,i] <- c(VMD4.Pred.train[,i],VMD4.Pred.test[,i])
  
  # #Metrics
  VMD4.SMAPE <- smape(VMD4.Pred.test[,i], Obs.test)
  VMD4.RRMSE <- RMSE(VMD4.Pred.test[,i], Obs.test)/mean(VMD4.Pred.test[,i])
  VMD4.R2    <- cor(VMD4.Pred.test[,i], Obs.test)^2
  
  VMD4.SMAPE.train <- smape(VMD4.Pred.train[,i], Obs.train)
  VMD4.RRMSE.train <- RMSE(VMD4.Pred.train[,i], Obs.train)/mean(VMD4.Pred.train[,i])
  VMD4.R2.train    <- cor(VMD4.Pred.train[,i], Obs.train)^2
  
  VMD4.Metrics.train[i,] <- c(i, 
                                VMD4.SMAPE.train,
                                VMD4.RRMSE.train,
                                VMD4.R2.train)
  VMD4.Metrics[i,] <- c(i,
                          VMD4.SMAPE,
                          VMD4.RRMSE,
                          VMD4.R2)
}

xtable::xtable(VMD4.Metrics, digits = 4)

save.image("4step-final.RData")
}

# Stacking step-ahead Predictions-----------------------------------

## 2-step-ahead
{
# setwd(ResultsDir)
# load('2step-final.RData')

set.seed(1234)
stack2.database <- data.frame(Obs,Pred2step)
colnames(stack2.database) <- c("y","x1","x2","x3","x4")

n <- dim(stack2.database)[1]
cut <- 0.7 * n

stack2.database.train <- stack2.database[1:cut,]
stack2.database.test  <- tail(stack2.database,n-cut)

meta.list <- c("cubist")

preprocess.list <- c("corr","pca","BoxCox")

stack <- list()

for (p in 1:length(preprocess.list)) {
  stack[[p]] <- train(y~.,data = stack2.database.train,
                      method = meta.list,
                      trControl = control,
                      preProcess = preprocess.list[p],
                      trace = FALSE)
  cat("PP: ", preprocess.list[p], "\t", p/length(preprocess.list)*100, "%\n", sep = "")
}

{
  h <- 2
  obsTR <- matrix(seq(1,length(Obs.train),1),ncol = h, byrow = TRUE)
  obsTE <- matrix(seq(1,length(Obs.test)+1,1),ncol = h, byrow = TRUE)
  PTRmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.train))
  PTEmo     <- matrix(ncol = length(preprocess.list), nrow = length(Obs.test))
  colnames(PTRmo) <- preprocess.list
  colnames(PTEmo) <- preprocess.list
  stack2.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                        nrow = n)
  stack2.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                 ncol = 4)
  stack2.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                 ncol = 4)
  colnames(stack2.Metrics) <- c("m","SMAPE","RRMSE","R2")
  rownames(stack2.Metrics) <- preprocess.list
}

for (m in 1:length(preprocess.list)) {
  x_trainm <- as.data.frame(stack2.database.train[,-1])
  x_testm  <- as.data.frame(stack2.database.test[,-1])
  
  # Train
  for (N in 1:h) {
    for (v in 1:dim(obsTR)[1]) {
      p <- obsTR[v,N]
      if (p <= dim(x_trainm)[1]) {
        if(p==obsTR[v,1]){
          PTRmo[p,m] <- predict(stack[[m]],x_trainm[p,])
        }
        else {
          x_trainm[p,1] <- PTRmo[p-1,m]
          PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
        }
      }
      else {
        break
      }
    }
  }
  
  # Test
  for (N in 1:h) {
    for (v in 1:dim(obsTE)[1]) {
      p <- obsTE[v,N]
      if (p <= dim(x_testm)[1]) {
        if(p==obsTE[v,1]){
          PTEmo[p,m] <- predict(stack[[m]],x_testm[p,])
        }
        else {
          x_testm[p,1] <- PTEmo[p-1,m]
          PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
        }
      }
      else {
        break
      }
    }
  }
  
  stack2.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
  
  ### Metrics
  
  pred2.SMAPE.train <- smape(PTRmo[,m], Obs.train)
  pred2.RRMSE.train <- RMSE(PTRmo[,m], Obs.train)/mean(PTRmo[,m])
  pred2.R2.train    <- cor(PTRmo[,m], Obs.train)^2
  
  pred2.SMAPE <- smape(PTEmo[,m], Obs.test)
  pred2.RRMSE <- RMSE(PTEmo[,m], Obs.test)/mean(PTEmo[,m])
  pred2.R2    <- cor(PTEmo[,m], Obs.test)^2
  
  
  stack2.Metrics.train[m,] <- c(m,pred2.SMAPE.train,pred2.RRMSE.train,pred2.R2.train)
  stack2.Metrics[m,] <- c(m,pred2.SMAPE,pred2.RRMSE,pred2.R2)
  
  cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
}

xtable::xtable(stack2.Metrics, digits = 4)

# save.image("2step-results-stack.RData")
}

## 3-step-ahead 
{
  # setwd(ResultsDir)
  # load('3step-final.RData')
  
  set.seed(1234)
  stack3.database <- data.frame(Obs,VMD3.Prediction)
  colnames(stack3.database) <- c("y","x1","x2","x3","x4")
  
  n <- dim(stack3.database)[1]
  cut <- 0.7 * n
  
  stack3.database.train <- stack3.database[1:cut,]
  stack3.database.test  <- tail(stack3.database,n-cut)
  
  meta.list <- c("cubist")
  
  preprocess.list <- c("corr","pca","BoxCox")
  
  stack <- list()
  
  for (p in 1:length(preprocess.list)) {
    stack[[p]] <- train(y~.,data = stack3.database.train,
                        method = meta.list,
                        trControl = control,
                        preProcess = preprocess.list[p],
                        trace = FALSE)
    cat("PP: ", preprocess.list[p], "\t", p/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  {
    h <- 3
    obsTR <- matrix(seq(1,length(Obs.train)+2,1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,length(Obs.test)+2,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.train))
    PTEmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.test))
    colnames(PTRmo) <- preprocess.list
    colnames(PTEmo) <- preprocess.list
    stack3.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                          nrow = n)
    stack3.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    stack3.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    colnames(stack3.Metrics) <- c("m","SMAPE","RRMSE","R2")
    rownames(stack3.Metrics) <- preprocess.list
  }
  
  for (m in 1:length(preprocess.list)) {
    x_trainm <- as.data.frame(stack3.database.train[,-1])
    x_testm  <- as.data.frame(stack3.database.test[,-1])
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR)[1]) {
        p <- obsTR[v,N]
        if (p <= dim(x_trainm)[1]) {
          if(p==obsTR[v,1]) {
            PTRmo[p,m] <- predict(stack[[m]],x_trainm[p,])
          }
          else if(p==obsTR[v,2]) {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            x_trainm[p,2] <- x_trainm[p-1,1]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    # Test
    for (N in 1:h) {
      for (v in 1:dim(obsTE)[1]) {
        p <- obsTE[v,N]
        if (p <= dim(x_testm)[1]) {
          if(p==obsTE[v,1]) {
            PTEmo[p,m] <- predict(stack[[m]],x_testm[p,])
          }
          else if(p==obsTE[v,2]) {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            x_testm[p,2] <- x_testm[p-1,1]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    stack3.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred3.SMAPE.train <- smape(PTRmo[,m], Obs.train)
    pred3.RRMSE.train <- RMSE(PTRmo[,m], Obs.train)/mean(PTRmo[,m])
    pred3.R2.train    <- cor(PTRmo[,m], Obs.train)^2
    
    pred3.SMAPE <- smape(PTEmo[,m], Obs.test)
    pred3.RRMSE <- RMSE(PTEmo[,m], Obs.test)/mean(PTEmo[,m])
    pred3.R2    <- cor(PTEmo[,m], Obs.test)^2
    
    
    stack3.Metrics.train[m,] <- c(m,pred3.SMAPE.train,pred3.RRMSE.train,pred3.R2.train)
    stack3.Metrics[m,] <- c(m,pred3.SMAPE,pred3.RRMSE,pred3.R2)
    
    cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
    
    save.image("3step-results-stack.RData")
  }
  
  xtable::xtable(stack3.Metrics, digits = 4)
  
}

## 4-step-ahead 
{
  # setwd(ResultsDir)
  # load('4step-final.RData')
  
  set.seed(1234)
  stack4.database <- data.frame(Obs,VMD4.Prediction)
  colnames(stack4.database) <- c("y","x1","x2","x3","x4")
  
  n <- dim(stack4.database)[1]
  cut <- 0.7 * n
  
  stack4.database.train <- stack4.database[1:cut,]
  stack4.database.test  <- tail(stack4.database,n-cut)
  
  meta.list <- c("cubist")
  
  preprocess.list <- c("corr","pca","BoxCox")
  
  stack <- list()
  
  for (p in 1:length(preprocess.list)) {
    stack[[p]] <- train(y~.,data = stack4.database.train,
                        method = meta.list,
                        trControl = control,
                        preProcess = preprocess.list[p],
                        trace = FALSE)
    cat("PP: ", preprocess.list[p], "\t", p/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  {
    h <- 4
    obsTR <- matrix(seq(1,length(Obs.train),1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,length(Obs.test)+3,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.train))
    PTEmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.test))
    colnames(PTRmo) <- preprocess.list
    colnames(PTEmo) <- preprocess.list
    stack4.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                          nrow = n)
    stack4.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    stack4.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    colnames(stack4.Metrics) <- c("m","SMAPE","RRMSE","R2")
    rownames(stack4.Metrics) <- preprocess.list
  }
  
  for (m in 1:length(preprocess.list)) {
    x_trainm <- as.data.frame(stack4.database.train[,-1])
    x_testm  <- as.data.frame(stack4.database.test[,-1])
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR)[1]) {
        p <- obsTR[v,N]
        if (p <= dim(x_trainm)[1]) {
          if(p==obsTR[v,1]) {
            PTRmo[p,m] <- predict(stack[[m]],x_trainm[p,])
          }
          else if(p==obsTR[v,2]) {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
          else if(p==obsTR[v,3]) {
            x_trainm[p,1] <- PTRmo[p-1,m]
            x_trainm[p,2] <- x_trainm[p-1,1]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            x_trainm[p,2] <- x_trainm[p-1,1]
            x_trainm[p,3] <- x_trainm[p-1,2]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    # Test
    for (N in 1:h) {
      for (v in 1:dim(obsTE)[1]) {
        p <- obsTE[v,N]
        if (p <= dim(x_testm)[1]) {
          if(p==obsTE[v,1]) {
            PTEmo[p,m] <- predict(stack[[m]],x_testm[p,])
          }
          else if(p==obsTE[v,2]) {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
          else if(p==obsTE[v,3]) {
            x_testm[p,1] <- PTEmo[p-1,m]
            x_testm[p,2] <- x_testm[p-1,1]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            x_testm[p,2] <- x_testm[p-1,1]
            x_testm[p,3] <- x_testm[p-1,2]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    stack4.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred4.SMAPE.train <- smape(PTRmo[,m], Obs.train)
    pred4.RRMSE.train <- RMSE(PTRmo[,m], Obs.train)/mean(PTRmo[,m])
    pred4.R2.train    <- cor(PTRmo[,m], Obs.train)^2
    
    pred4.SMAPE <- smape(PTEmo[,m], Obs.test)
    pred4.RRMSE <- RMSE(PTEmo[,m], Obs.test)/mean(PTEmo[,m])
    pred4.R2    <- cor(PTEmo[,m], Obs.test)^2
    
    
    stack4.Metrics.train[m,] <- c(m,pred4.SMAPE.train,pred4.RRMSE.train,pred4.R2.train)
    stack4.Metrics[m,] <- c(m,pred4.SMAPE,pred4.RRMSE,pred4.R2)
    
    cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
    
    save.image("4step-results-stack.RData")
  }
  
  xtable::xtable(stack4.Metrics, digits = 4)
  
}



# Diebold-Mariano tests---------------------------------------------

colnames(PREDS1) <- c('(A) VMD-STACK-CORR','(B) VMD-STACK-PCA',
                      '(C) VMD-STACK-BOXCOX','(D) VMD-KNN','(E) VMD-SVR',
                      '(F) VMD-NNET','(G) VMD-GLM','(H) STACK-CORR',
                      '(I) STACK-PCA','(J) STACK-BOXCOX','(K) KNN',
                      '(L) SVR','(M) NNET','(N) GLM')

PREDS1 <- NULL

for (i in 1:3) {
  PREDS1 <- c(PREDS1,tail(stack.pred[,i+6],n-cut))
}
for (i in 1:4) {
  PREDS1 <- c(PREDS1,tail(VMD.Prediction[,i],n-cut))
}
for (i in 1:3) {
  PREDS1 <- c(PREDS1,tail(stack.pred[,i+6],n-cut))
}
for (i in 1:4) {
  PREDS1 <- c(PREDS1,tail(pred[,i],n-cut))
}

save(PREDS1, file = 'preds1.RData')


PREDS2 <- NULL

for (i in 1:3) {
  PREDS2 <- c(PREDS2,tail(stack2.pred[,i],n-cut))
}
for (i in 1:4) {
  PREDS2 <- c(PREDS2,tail(Pred2step[,i],n-cut))
}
for (i in 1:3) {
  PREDS2 <- c(PREDS2,tail(stack2.pred[,i],n-cut))
}
for (i in 1:4) {
  PREDS2 <- c(PREDS2,tail(Pred2step[,i],n-cut))
}

save(PREDS2, file = 'preds2.RData')

PREDS3 <- NULL

for (i in 1:3) {
  PREDS3 <- c(PREDS3,tail(stack3.pred[,i],n-cut))
}
for (i in 1:4) {
  PREDS3 <- c(PREDS3,tail(VMD3.Prediction[,i],n-cut))
}
for (i in 1:3) {
  PREDS3 <- c(PREDS3,tail(stack3.pred[,i],n-cut))
}
for (i in 1:4) {
  PREDS3 <- c(PREDS3,tail(Pred3step[,i],n-cut))
}

save(PREDS3, file = 'preds3.RData')

## 1 step
{
  h <- 1
  
  e <- matrix(ncol = 14, nrow = length(Obs.test))
  colnames(e) <- c('(A) VMD-STACK-CORR','(B) VMD-STACK-PCA',
                   '(C) VMD-STACK-BOXCOX','(D) VMD-KNN','(E) VMD-SVR',
                   '(F) VMD-NNET','(G) VMD-GLM','(H) STACK-CORR',
                   '(I) STACK-PCA','(J) STACK-BOXCOX','(K) KNN',
                   '(L) SVR','(M) NNET','(N) GLM')
  
  for (i in 1:dim(e)[2]) {
    e[,i] <- (Obs.test - PREDS1[,i])^2
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  colnames(DM.tvalue) <- c('(A)','(B)','(C)','(D)','(E)','(F)','(G)',
                           '(H)','(I)','(J)','(K)','(L)','(M)','(N)')
  rownames(DM.tvalue) <- c('(A) VMD-STACK-CORR','(B) VMD-STACK-PCA',
                           '(C) VMD-STACK-BOXCOX','(D) VMD-KNN','(E) VMD-SVR',
                           '(F) VMD-NNET','(G) VMD-GLM','(H) STACK-CORR',
                           '(I) STACK-PCA','(J) STACK-BOXCOX','(K) KNN',
                           '(L) SVR','(M) NNET','(N) GLM')
  colnames(DM.pvalue) <- c('(A)','(B)','(C)','(D)','(E)','(F)','(G)',
                           '(H)','(I)','(J)','(K)','(L)','(M)','(N)')
  rownames(DM.pvalue) <- c('(A) VMD-STACK-CORR','(B) VMD-STACK-PCA',
                           '(C) VMD-STACK-BOXCOX','(D) VMD-KNN','(E) VMD-SVR',
                           '(F) VMD-NNET','(G) VMD-GLM','(H) STACK-CORR',
                           '(I) STACK-PCA','(J) STACK-BOXCOX','(K) KNN',
                           '(L) SVR','(M) NNET','(N) GLM')
  
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

## 2 steps
{
  h <- 2
  
  e <- matrix(ncol = 14, nrow = length(Obs.test))
  colnames(e) <- c('(1) VMD-STACK-CORR','(2) VMD-STACK-PCA',
                   '(3) VMD-STACK-BOXCOX','(4) VMD-KNN','(5) VMD-SVR',
                   '(6) VMD-NNET','(7) VMD-GLM','(8) STACK-CORR',
                   '(9) STACK-PCA','(10) STACK-BOXCOX','(11) KNN',
                   '(12) SVR','(13) NNET','(14) GLM')
  
  for (i in 1:dim(e)[2]) {
    e[,i] <- (Obs.test - PREDS2[,i])^2
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  colnames(DM.tvalue) <- c('(1)','(2)','(3)','(4)','(5)','(6)','(7)',
                           '(8)','(9)','(10)','(11)','(12)','(13)','(14)')
  rownames(DM.tvalue) <- c('(1) VMD-STACK-CORR','(2) VMD-STACK-PCA',
                           '(3) VMD-STACK-BOXCOX','(4) VMD-KNN','(5) VMD-SVR',
                           '(6) VMD-NNET','(7) VMD-GLM','(8) STACK-CORR',
                           '(9) STACK-PCA','(10) STACK-BOXCOX','(11) KNN',
                           '(12) SVR','(13) NNET','(14) GLM')
  colnames(DM.pvalue) <- c('(1)','(2)','(3)','(4)','(5)','(6)','(7)',
                           '(8)','(9)','(10)','(11)','(12)','(13)','(14)')
  rownames(DM.pvalue) <- c('(1) VMD-STACK-CORR','(2) VMD-STACK-PCA',
                           '(3) VMD-STACK-BOXCOX','(4) VMD-KNN','(5) VMD-SVR',
                           '(6) VMD-NNET','(7) VMD-GLM','(8) STACK-CORR',
                           '(9) STACK-PCA','(10) STACK-BOXCOX','(11) KNN',
                           '(12) SVR','(13) NNET','(14) GLM')
  
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
  xtable::xtable(DM.tvalue, digits = 4)
  xtable::xtable(DM.pvalue, digits = 4)
}

## 3 steps
{
  h <- 3
  
  e <- matrix(ncol = 14, nrow = length(Obs.test))
  colnames(e) <- c('(1) VMD-STACK-CORR','(2) VMD-STACK-PCA',
                   '(3) VMD-STACK-BOXCOX','(4) VMD-KNN','(5) VMD-SVR',
                   '(6) VMD-NNET','(7) VMD-GLM','(8) STACK-CORR',
                   '(9) STACK-PCA','(10) STACK-BOXCOX','(11) KNN',
                   '(12) SVR','(13) NNET','(14) GLM')
  
  for (i in 1:dim(e)[2]) {
    e[,i] <- (Obs.test - PREDS3[,i])
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = dim(e)[2])
  colnames(DM.tvalue) <- c('(1)','(2)','(3)','(4)','(5)','(6)','(7)',
                           '(8)','(9)','(10)','(11)','(12)','(13)','(14)')
  rownames(DM.tvalue) <- c('(1) VMD-STACK-CORR','(2) VMD-STACK-PCA',
                           '(3) VMD-STACK-BOXCOX','(4) VMD-KNN','(5) VMD-SVR',
                           '(6) VMD-NNET','(7) VMD-GLM','(8) STACK-CORR',
                           '(9) STACK-PCA','(10) STACK-BOXCOX','(11) KNN',
                           '(12) SVR','(13) NNET','(14) GLM')
  colnames(DM.pvalue) <- c('(1)','(2)','(3)','(4)','(5)','(6)','(7)',
                           '(8)','(9)','(10)','(11)','(12)','(13)','(14)')
  rownames(DM.pvalue) <- c('(1) VMD-STACK-CORR','(2) VMD-STACK-PCA',
                           '(3) VMD-STACK-BOXCOX','(4) VMD-KNN','(5) VMD-SVR',
                           '(6) VMD-NNET','(7) VMD-GLM','(8) STACK-CORR',
                           '(9) STACK-PCA','(10) STACK-BOXCOX','(11) KNN',
                           '(12) SVR','(13) NNET','(14) GLM')
  
  for (i in 1:3) {
    for (j in 1:dim(e)[2]) {
      if(i>=j) {
        DM.tvalue[j,i] <- NA
        DM.pvalue[j,i] <- NA
      }
      else {
        DMtest <- dm.test(e[,i],e[,j], h = h, power = 1)
        DM.tvalue[j,i] <- DMtest$statistic
        DM.pvalue[j,i] <- DMtest$p.value
      }
    }
  }
  xtable::xtable(DM.tvalue, digits = 4)
  xtable::xtable(DM.pvalue, digits = 4)
}




# Plots and TEMP----------------------------------------------------

load('SE1.RData')
boxplot_error(Obs.test,PREDS1)
load('SE2.RData')
boxplot_error(Obs.test,PREDS2)
load('SE3.RData')
boxplot_error(Obs.test,PREDS3)

preds.se <- matrix(ncol = 4, nrow = length(PREDS2))

preds.se[,1] <- rep(Obs.test, times = 14)

preds.se[,2] <- PREDS1[-c(1,
                          615,
                          1229,
                          1843,
                          2457,
                          3071,
                          3685,
                          4299,
                          4913,
                          5527,
                          6141,
                          6755,
                          7369,
                          7983)]

preds.se[,3] <- PREDS2
preds.se[,4] <- PREDS3

colnames(preds.se) <- c('Obs','One','Two','Three')

models <- c('(A)','(B)','(C)','(D)','(E)','(F)','(G)',
            '(H)','(I)','(J)','(K)','(L)','(M)','(N)')

preds.se <- as.data.frame(preds.se)

preds.se$APE3 <- abs(preds.se$Obs-preds.se$Three)/preds.se$Obs

data <- data.frame(as.vector(unlist(data.frame(preds.se$APE1,
                                               preds.se$APE2,
                                               preds.se$APE3))),
                   rep(c('One','Two','Three'),each = length(PREDS2)),
                   rep(models, each = length(Obs.test)))

colnames(data) <- c('APE','Forecast','Models')

source('boxplot_error.R')
load('datase.RData')
source('bp_error.R')

bp_error(data)

setwd(BaseDir)


Plot2(p$date,p$Obs,p$Pred1,p$Pred2,p$Pred3)
Plot_Recorte(p$date,p$Obs,p$Pred1,p$Pred2,p$Pred3)
plot_IMF(dataset$Date,dataset$Close,IMF[,1],IMF[,2],IMF[,3],IMF[,4],IMF[,5])
