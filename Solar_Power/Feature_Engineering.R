FE<-function(data)
{
  cat("\014")  
  library(moments)
  options(warn=-1)
  meanstra <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  sdtra    <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  skewtra  <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  diftra   <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  expo2    <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  expo3    <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  ftanh    <-matrix(nrow=dim(data)[1],ncol=dim(data)[2]) 
  flog     <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  min      <-matrix(nrow=dim(data)[1],ncol=dim(data)[2]) 
  max      <-matrix(nrow=dim(data)[1],ncol=dim(data)[2])
  type<-is.data.frame(data)
  
  if (type == FALSE)
  {
    cat('Data set must be a data.frame')
  }
  else
  {
    cat('Feature Engineering process start! \n')
    
    for(j in 1:dim(data)[2])    #Columns
    {
      for (i in 1:dim(data)[1]) #Lines
      { 
        meanstra[i,j]<-ifelse(i==1,mean(c(data[1,j],data[1,j],data[1,j])),
                              ifelse(i==2,mean(c(data[1,j],data[1,j],data[2,j])),
                                     mean(c(data[i,j],data[(i-1),j],data[(i-2),j]))))
                              
        sdtra[i,j]<-ifelse(i==1,sd(c(data[1,j],data[1,j],data[1,j])),
                           ifelse(i==2,sd(c(data[1,j],data[1,j],data[2,j])),
                                  sd(c(data[i,j],data[(i-1),j],data[(i-2),j]))))
        
        skewtra[i,j]<-ifelse(i==1,skewness(c(data[1,j],data[1,j],data[1,j])),
                             ifelse(i==2,skewness(c(data[1,j],data[1,j],data[2,j])),
                                    skewness(c(data[i,j],data[(i-1),j],data[(i-2),j]))))
        
        diftra[i,j]<-ifelse(i==1,data[1,j]-data[1,j],data[i,j]-data[i-1,j])
        

        min[i,j]<-ifelse(i==1,min(c(data[1,j],data[1,j],data[1,j])),
                         ifelse(i==2,min(c(data[1,j],data[1,j],data[2,j])),
                                min(c(data[i,j],data[(i-1),j],data[(i-2),j]))))
        
        max[i,j]<-ifelse(i==1,max(c(data[1,j],data[1,j],data[1,j])),
                         ifelse(i==2,max(c(data[1,j],data[1,j],data[2,j])),
                                max(c(data[i,j],data[(i-1),j],data[(i-2),j]))))
        
      }
      
    expo2[,j]<-(data[,j])^2

    expo3[,j]<-(data[,j])^3

    ftanh[,j]<-tanh(data[,j])

    flog[,j]<-log(data[,j])

    skewtra[is.na(skewtra[,j]), j]  <- mean(skewtra[,j], na.rm = TRUE)
    
    cat("Feature build to colum",j,"of", dim(data)[2],"\n")
    }
  
    Results <-data.frame(meanstra,sdtra,skewtra,diftra,expo2,expo3,ftanh,flog,min,max)
    Results <-Filter(function(x) sd(x) != 0, Results)
    
    cat("Feature Engineering process done!\n")
    Features<-list(Results)
    
    # print(Features)
  }
   
}


