PO3<-function(Obs,Pred1,Pred2,Pred3,Pred4,Pred5,Pred6,
              nameM1,nameM2,nameM3,nameM4,nameM5,nameM6)
{
  n <- dim(Obs)[1]
  cut <- 0.7*n
  
  data1<-data.frame(rep(c("M1","M2","M3","M4","M5","M6"), each = dim(Obs)[1]),
                    rep(Obs),
                    as.vector(unlist(data.frame(Pred1,Pred2,Pred3,Pred4,Pred5,Pred6))),
                    rep(c("Training","Test"), 
                        times = c(length(Obs[1:cut,]),dim(tail(Obs,n-cut))[1])),
                    rep(seq(1,dim(Obs)[1])))
  colnames(data1)<-c("Model","Observed","Predicted","Type","Frequency")
  
  Names <- c(M1 = nameM1,
             M2 = nameM2,
             M3 = nameM3,
             M4 = nameM4,
             M5 = nameM5,
             M6 = nameM6)
  
  g2 <- ggplot(data1, aes(x=Observed, y=Predicted))
  g2 <- g2 + geom_point(aes(shape=Type, color = Type),size=3)
  g2 <- g2 + geom_abline(slope = 1, intercept = 0)
  g2 <- g2 + scale_color_manual(values=c("#000000","#FF0000"))
  g2 <- g2 + xlab("Observed") + ylab("Predicted") + ggtitle('')
  g2 <- g2 + facet_wrap(~Model,scales="free",labeller = labeller(Model = Names))
  g2 <- g2 + theme_bw(base_size = 18)
  g2 <- g2 + theme(legend.position = "bottom", 
                   legend.title = element_blank(),
                   axis.text=element_text(size=18),
                   # legend.text=element_text(size=rel(1)),
                   plot.title = element_text(hjust=0.5),
                   # text=element_text(family="Times New Roman"),
                   axis.title=element_text(size=18))
  print(g2)
}



# PO4<-function(Obs,Pred1,Pred2,Pred3)
# {
#   data1<-data.frame(rep(c("rf","gbm","knn"), each = length(y_teste)),
#                     rep(y_teste),
#                     as.vector(unlist(data.frame(pred_svmLinear2,pred_brnn,pred_bridge))),
#                     rep("Test", times=length(y_teste)),
#                     rep(seq(1,length(y_teste))))
#   colnames(data1)<-c("Model","Observed","Predicted","Type","Frequency")
#   
#   g2 <- ggplot(data1, aes(x=Observed, y=Predicted))
#   g2 <- g2 + geom_point(aes(shape=Type, color = Type),size=3)
#   g2 <- g2 + geom_abline(slope = 1, intercept = 0)
#   g2 <- g2 + scale_color_manual(values=c("#000000","#FF0000"))
#   g2 <- g2 + xlab("Observed") + ylab("Predicted") + ggtitle('')
#   g2 <- g2 + facet_wrap(~Model,scales="free")
#   g2 <- g2 + theme_bw(base_size = 18)
#   g2 <- g2 + theme(legend.position = "bottom", 
#                    legend.title = element_blank(),
#                    axis.text=element_text(size=18),
#                    legend.text=element_text(size=rel(1)),
#                    plot.title = element_text(hjust=0.5),
#                    text=element_text(family="Times New Roman"),
#                    axis.title=element_text(size=12))
#   print(g2)
# }

# "rf","gbm","knn"
# "svmLinear2","brnn","bridge"

# as.vector(unlist(data.frame(Obs,Pred1,Pred2,Pred3,Pred4,
#                             Pred5,Pred6,Pred7)))
# 
# data<-data.frame(as.vector(unlist(data.frame(Obs,Pred))),
#                  rep(c("y(t) - Estimated","y(t)-Predicted"),each=dim(Obs)[1]),
#                  rep(seq(1,dim(Obs)[1]),times=2))
# 
# data1<-data.frame(rep(c("Stack","rf","gbm","knn","svmLinear2","brnn","bridge"), each = dim(Obs)[1]),
#                   rep(Obs),
#                   as.vector(unlist(data.frame(Pred_stack,Pred_rf,Pred_gbm,Pred_knn,
#                                               Pred_svmLinear2,Pred_brnn,Pred_bridge))),
#                   rep(c("Training","Test"), 
#                       times = c(round(0.7*dim(Obs)[1]),round(0.3*dim(Obs)[1]))),
#                   rep(seq(1,dim(Obs)[1])))
