# Starting Code ----------------------------------------------------
## Clear memory
rm(list=ls(all=TRUE))

## Set working directory
setwd("~/Doc/WCCI/")

BaseDir       <- getwd()
ResultsDir    <- paste(BaseDir, "Results-Stack", sep="/")

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

## Cores Cluster
# ncl <- detectCores();ncl
# cl  <- makeCluster(ncl-1);registerDoParallel(cl)
# stopImplicitCluster() # Stop

# Data treatment ---------------------------------------------------

## Load data

url <- "https://raw.githubusercontent.com/ByronKKing/Time-Series-R/master/bitcoin-2016-02-21.csv"

dataset <- read.csv(url, sep = ",")

## dataframes

lag <- 1

dataset.lag <- data.frame(dataset[,5][(lag+1):(dim(dataset)[1])],
                          dataset[,2][lag:(dim(dataset)[1]-lag)],
                          dataset[,3][lag:(dim(dataset)[1]-lag)],
                          dataset[,4][lag:(dim(dataset)[1]-lag)])

colnames(dataset.lag) <- c('y','Open','High','Low')

n <- dim(dataset.lag)[1]
cut <- 0.7 * n

train <- dataset.lag[1:cut,]
test <- tail(dataset.lag,n-cut)

x_train <- train[,-1]
y_train <- train[,1]

x_test <- test[,-1]
y_test <- test[,1]

setwd(ResultsDir)
save.image("lag-data.RData")

# Training and Predictions------------------------------------------
setwd(ResultsDir)
load("lag-data.RData")

set.seed(1234)

control <- trainControl(method="cv", 
                        number=5, 
                        savePredictions='final',
                        verboseIter = FALSE)

model.list <- c('knn', 
                'svmLinear2', 
                'nnet',
                'glm') 

{
  model <- list()
  pred.train <- list()
  pred.test  <- list()
  pred <- matrix(ncol = length(model.list), nrow = n)
  
  Metrics.train <- matrix(nrow = length(model.list),ncol = 6)
  Metrics       <- matrix(nrow = length(model.list),ncol = 6)
  colnames(Metrics) <- c("i","SMAPE","MASE","RRMSE","MAPE","R2")
  rownames(Metrics) <- model.list
}

for (i in 1:length(model.list)) {
  model[[i]] <- train(y~., data = train,
                      preProcess = c("BoxCox"),
                      method = model.list[i],
                      trControl = control,
                      tuneLength = 5,
                      trace = FALSE)
  
  pred.train[[i]] <- predict(model[[i]],train)
  pred.test[[i]]  <- predict(model[[i]],test)
  pred[,i]   <- c(pred.train[[i]],pred.test[[i]])
  
  
  SMAPE <- smape(pred.test[[i]], y_test)
  MASE  <- mase(pred.test[[i]], y_test)
  RRMSE <- RMSE(pred.test[[i]], y_test)/mean(pred.test[[i]])
  MAPE  <- mape(pred.test[[i]], y_test)
  R2    <- cor(pred.test[[i]], y_test)^2
  
  SMAPE.train <- smape(pred.train[[i]], y_train)
  MASE.train  <- mase(pred.train[[i]], y_train)
  RRMSE.train <- RMSE(pred.train[[i]], y_train)/mean(pred.train[[i]])
  MAPE.train  <- mape(pred.train[[i]], y_train)
  R2.train    <- cor(pred.train[[i]], y_train)^2
  
  Metrics.train[i,] <- c(i, SMAPE.train,MASE.train,RRMSE.train,
                         MAPE.train,R2.train)
  Metrics[i,] <- c(i,SMAPE,MASE,RRMSE,MAPE,R2)
  
  cat("\nModel:", model.list[i],"SMAPE:", SMAPE, 
      "RRMSE:", RRMSE,"R2:", R2, as.character(Sys.time()),"\n")
  save.image("Results-individual.RData")
}

xtable::xtable(Metrics[,c(2,4,6)], digits = 4)


# Stacking Training and Predictions---------------------------------
# setwd(ResultsDir)
# load("Results-individual.RData")
set.seed(1234)
stack.database <- data.frame(dataset.lag$y,pred)
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
    
    
    stack.SMAPE <- smape(stack.pred.test, test$y)
    stack.RRMSE <- RMSE(stack.pred.test, test$y)/mean(stack.pred.test)
    stack.R2    <- cor(stack.pred.test, test$y)^2
    
    stack.SMAPE.train <- smape(stack.pred.train, train$y)
    stack.RRMSE.train <- RMSE(stack.pred.train, train$y)/mean(stack.pred.train)
    stack.R2.train    <- cor(stack.pred.train, train$y)^2
    
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
  save.image("stack-results.RData")
}

stack.Metrics

xtable::xtable(stack.Metrics[c(7:9),], digits = 4)

# Step-ahead Predictions--------------------------------------------
setwd(ResultsDir)
load("Results-individual.RData")

## 2-step-ahead
{
  
  {
    lag <- 2
    
    dataset2.lag <- data.frame(dataset[,5][(lag+1):(dim(dataset)[1])],
                               dataset[,5][(lag-0):(dim(dataset)[1]-lag+1)],
                               dataset[,5][(lag-1):(dim(dataset)[1]-lag+0)],
                               dataset[,2][(lag-1):(dim(dataset)[1]-lag+0)],
                               dataset[,3][(lag-1):(dim(dataset)[1]-lag+0)],
                               dataset[,4][(lag-1):(dim(dataset)[1]-lag+0)])
    
    colnames(dataset2.lag) <- c('y(t)','y(t-1)','y(t-2)',
                                'Open','High','Low')
    
    n <- dim(dataset2.lag)[1]
    cut <- 0.7 * n
    
    train2 <- dataset2.lag[1:cut,]
    test2 <- tail(dataset2.lag,n-cut)
    
    x_train2 <- train2[,-1]
    y_train2 <- train2[,1]
    
    x_test2 <- test2[,-1]
    y_test2 <- test2[,1]
  }
  
  {
    h <- 2
    obsTR <- matrix(seq(1,dim(x_train2)[1],1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,dim(x_test2)[1]+1,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(model.list), nrow = dim(train2)[1])
    PTEmo     <- matrix(ncol = length(model.list), nrow = dim(test2)[1])
    Pred2step <- matrix(ncol = length(model.list), nrow = n)
    colnames(PTRmo) <- model.list
    colnames(PTEmo) <- model.list
    Metrics2.train <- matrix(nrow = length(model.list), ncol = 4)
    Metrics2       <- matrix(nrow = length(model.list), ncol = 4)
    colnames(Metrics2.train) <- c("m","SMAPE","RRMSE","R2") 
    colnames(Metrics2)       <- c("m","SMAPE","RRMSE","R2")
    rownames(Metrics2)       <- model.list
  }
  
  for (m in 1:length(model.list)) {
    x_trainm <- as.data.frame(x_train2)
    x_testm  <- as.data.frame(x_test2)
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR)[1]) {
        p <- obsTR[v,N]
        if (p <= dim(x_trainm)[1]) {
          if(p==obsTR[v,1]){
            PTRmo[p,m] <- predict(model[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(model[[m]],x_trainm[p,])
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
            PTEmo[p,m] <- predict(model[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(model[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    ### Avoiding negative values
    for (j in 1:dim(PTRmo)[1]) {
      if (PTRmo[j,m] < 0) {
        PTRmo[j,m] <- 0
      }
    }
    for (j in 1:dim(PTEmo)[1]) {
      if (PTEmo[j,m] < 0) {
        PTEmo[j,m] <- 0
      }
    }
    
    Pred2step[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred2.SMAPE.train <- smape(PTRmo[,m], train2[,1])
    pred2.RRMSE.train <- RMSE(PTRmo[,m], train2[,1])/mean(PTRmo[,m])
    pred2.R2.train    <- cor(PTRmo[,m], train2[,1])^2
    
    pred2.SMAPE <- smape(PTEmo[,m], test2[,1])
    pred2.RRMSE <- RMSE(PTEmo[,m], test2[,1])/mean(PTEmo[,m])
    pred2.R2    <- cor(PTEmo[,m], test2[,1])^2
    
    
    Metrics2.train[m,] <- c(m,pred2.SMAPE.train,pred2.RRMSE.train,pred2.R2.train)
    Metrics2[m,] <- c(m,pred2.SMAPE,pred2.RRMSE,pred2.R2)
    
    cat("Model: ", model.list[m]," ",m/length(model.list)*100, "%\n", sep = "")
  }
  
  xtable::xtable(Metrics2, digits = 4)
  
  save.image("2step-results.RData")
}

## 3-step-ahead
{

{
  lag <- 3
  
  dataset3.lag <- data.frame(dataset[,5][(lag+1):(dim(dataset)[1])],
                             dataset[,5][(lag-0):(dim(dataset)[1]-lag+2)],
                             dataset[,5][(lag-1):(dim(dataset)[1]-lag+1)],
                             dataset[,5][(lag-2):(dim(dataset)[1]-lag+0)],
                             dataset[,2][(lag-2):(dim(dataset)[1]-lag+0)],
                             dataset[,3][(lag-2):(dim(dataset)[1]-lag+0)],
                             dataset[,4][(lag-2):(dim(dataset)[1]-lag+0)])
  
  colnames(dataset3.lag) <- c('y(t)','y(t-1)','y(t-2)','y(t-3)',
                              'Open','High','Low')
  
  n <- dim(dataset3.lag)[1]
  cut <- 0.7 * n
  
  train3 <- dataset3.lag[1:cut,]
  test3 <- tail(dataset3.lag,n-cut)
  
  x_train3 <- train3[,-1]
  y_train3 <- train3[,1]
  
  x_test3 <- test3[,-1]
  y_test3 <- test3[,1]
}

{
  h <- 3
  obsTR <- matrix(seq(1,dim(x_train3)[1]+2,1),ncol = h, byrow = TRUE)
  obsTE <- matrix(seq(1,dim(x_test3)[1]+2,1),ncol = h, byrow = TRUE)
  PTRmo <- matrix(ncol = length(model.list), nrow = dim(train3)[1])
  PTEmo     <- matrix(ncol = length(model.list), nrow = dim(test3)[1])
  Pred3step <- matrix(ncol = length(model.list), nrow = n)
  colnames(PTRmo) <- model.list
  colnames(PTEmo) <- model.list
  Metrics3.train <- matrix(nrow = length(model.list), ncol = 4)
  Metrics3       <- matrix(nrow = length(model.list), ncol = 4)
  colnames(Metrics3.train) <- c("m","SMAPE","RRMSE","R2") 
  colnames(Metrics3)       <- c("m","SMAPE","RRMSE","R2")
  rownames(Metrics3)       <- model.list
}

for (m in 1:length(model.list)) {
  x_trainm <- as.data.frame(x_train3)
  x_testm  <- as.data.frame(x_test3)
  
  # Train
  for (N in 1:h) {
    for (v in 1:dim(obsTR)[1]) {
      p <- obsTR[v,N]
      if (p <= dim(x_trainm)[1]) {
        if(p==obsTR[v,1]){
          PTRmo[p,m] <- predict(model[[m]],x_trainm[p,])
        }
        else if(p==obsTR[v,2]) {
          x_trainm[p,1] <- PTRmo[p-1,m]
          PTRmo[p,m]    <- predict(model[[m]],x_trainm[p,])
        }
        else {
          x_trainm[p,1] <- PTRmo[p-1,m]
          x_trainm[p,2] <- x_trainm[p-1,1]
          
          PTRmo[p,m]   <-predict(model[[m]],x_trainm[p,])
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
          PTEmo[p,m] <- predict(model[[m]],x_testm[p,])
        }
        else if(p==obsTE[v,2]) {
          x_testm[p,1] <- PTEmo[p-1,m]
          PTEmo[p,m]    <- predict(model[[m]],x_testm[p,])
        }
        else {
          x_testm[p,1] <- PTEmo[p-1,m]
          x_testm[p,2] <- x_testm[p-1,1]
          
          PTEmo[p,m]   <-predict(model[[m]],x_testm[p,])
        }
      }
      else {
        break
      }
    }
  }
  
  ### Avoiding negative values
  for (j in 1:dim(PTRmo)[1]) {
    if (PTRmo[j,m] < 0) {
      PTRmo[j,m] <- 0
    }
  }
  for (j in 1:dim(PTEmo)[1]) {
    if (PTEmo[j,m] < 0) {
      PTEmo[j,m] <- 0
    }
  }
  
  Pred3step[,m] <- c(PTRmo[,m],PTEmo[,m])
  
  ### Metrics
  
  pred3.SMAPE.train <- smape(PTRmo[,m], train3[,1])
  pred3.RRMSE.train <- RMSE(PTRmo[,m], train3[,1])/mean(PTRmo[,m])
  pred3.R2.train    <- cor(PTRmo[,m], train3[,1])^2
  
  pred3.SMAPE <- smape(PTEmo[,m], test3[,1])
  pred3.RRMSE <- RMSE(PTEmo[,m], test3[,1])/mean(PTEmo[,m])
  pred3.R2    <- cor(PTEmo[,m], test3[,1])^2
  
  
  Metrics3.train[m,] <- c(m,pred3.SMAPE.train,pred3.RRMSE.train,pred3.R2.train)
  Metrics3[m,] <- c(m,pred3.SMAPE,pred3.RRMSE,pred3.R2)
  
  cat("Model: ", model.list[m]," ",m/length(model.list)*100, "%\n", sep = "")
}

xtable::xtable(Metrics3, digits = 4)

save.image("3step-results.RData")
}

# Stacking step-ahead Predictions-----------------------------------

## 2-step-ahead
{
  # setwd(ResultsDir)
  # load('2step-results.RData')
  
  set.seed(1234)
  stack2.database <- data.frame(dataset2.lag[,1],Pred2step)
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
    obsTR <- matrix(seq(1,dim(train2)[1],1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,dim(test2)[1]+1,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = dim(train2)[1])
    PTEmo     <- matrix(ncol = length(preprocess.list), nrow = dim(test2)[1])
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
    
    ### Avoiding negative values
    for (j in 1:dim(PTRmo)[1]) {
      if (PTRmo[j,m] < 0) {
        PTRmo[j,m] <- 0
      }
    }
    for (j in 1:dim(PTEmo)[1]) {
      if (PTEmo[j,m] < 0) {
        PTEmo[j,m] <- 0
      }
    }
    
    stack2.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred2.SMAPE.train <- smape(PTRmo[,m], train2[,1])
    pred2.RRMSE.train <- RMSE(PTRmo[,m], train2[,1])/mean(PTRmo[,m])
    pred2.R2.train    <- cor(PTRmo[,m], train2[,1])^2
    
    pred2.SMAPE <- smape(PTEmo[,m], test2[,1])
    pred2.RRMSE <- RMSE(PTEmo[,m], test2[,1])/mean(PTEmo[,m])
    pred2.R2    <- cor(PTEmo[,m], test2[,1])^2
    
    
    stack2.Metrics.train[m,] <- c(m,pred2.SMAPE.train,pred2.RRMSE.train,pred2.R2.train)
    stack2.Metrics[m,] <- c(m,pred2.SMAPE,pred2.RRMSE,pred2.R2)
    
    cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  xtable::xtable(stack2.Metrics, digits = 4)
  
  save.image("2step-results-stack.RData")
}

## 3-step-ahead 
{
  # setwd(ResultsDir)
  # load('3step-results.RData')
  
  set.seed(1234)
  stack3.database <- data.frame(dataset3.lag[,1],Pred3step)
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
    obsTR <- matrix(seq(1,dim(train3)[1]+2,1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,dim(test3)[1]+2,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = dim(train3)[1])
    PTEmo <- matrix(ncol = length(preprocess.list), nrow = dim(test3)[1])
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
    
    ### Avoiding negative values
    for (j in 1:dim(PTRmo)[1]) {
      if (PTRmo[j,m] < 0) {
        PTRmo[j,m] <- 0
      }
    }
    for (j in 1:dim(PTEmo)[1]) {
      if (PTEmo[j,m] < 0) {
        PTEmo[j,m] <- 0
      }
    }
    
    stack3.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred3.SMAPE.train <- smape(PTRmo[,m], train3[,1])
    pred3.RRMSE.train <- RMSE(PTRmo[,m], train3[,1])/mean(PTRmo[,m])
    pred3.R2.train    <- cor(PTRmo[,m], train3[,1])^2
    
    pred3.SMAPE <- smape(PTEmo[,m], test3[,1])
    pred3.RRMSE <- RMSE(PTEmo[,m], test3[,1])/mean(PTEmo[,m])
    pred3.R2    <- cor(PTEmo[,m], test3[,1])^2
    
    
    stack3.Metrics.train[m,] <- c(m,pred3.SMAPE.train,pred3.RRMSE.train,pred3.R2.train)
    stack3.Metrics[m,] <- c(m,pred3.SMAPE,pred3.RRMSE,pred3.R2)
    
    cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
    
    save.image("3step-results-stack.RData")
  }
  
  xtable::xtable(stack3.Metrics, digits = 4)
  
}
