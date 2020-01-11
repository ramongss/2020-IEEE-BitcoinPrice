PO9<-function(Obs,Pred1,Pred2,Pred3,Pred4,Pred5,Pred6,Pred7,Pred8,Pred9)
{
  data1<-data.frame(rep(c("M1","M2","M3",
                          "M4","M5","M6",
                          "M7","M8","M9"), each = dim(Obs)[1]),
                    rep(Obs),
                    as.vector(unlist(data.frame(Pred1,Pred2,Pred3,
                                                Pred4,Pred5,Pred6,
                                                Pred7,Pred8,Pred9))),
                    rep(c("Training","Test"), 
                        times = c(round(0.7*dim(Obs)[1]),round(0.3*dim(Obs)[1]))),
                    rep(seq(1,dim(Obs)[1])))
  colnames(data1)<-c("Model","Observed","Predicted","Type","Frequency")
  
  Names <- c(M1 = "XGBoost - 1",
             M2 = "XGBoost - 2",
             M3 = "XGBoost - 3",
             M4 = "Cubist - 1",
             M5 = "Cubist - 2",
             M6 = "Cubist - 3",
             M7 = "Random Forest - 1",
             M8 = "Random Forest - 2",
             M9 = "Random Forest - 3")
  
  g2 <- ggplot(data1, aes(x=Observed, y=Predicted))
  g2 <- g2 + geom_point(aes(shape=Type, color = Type),size=3)
  g2 <- g2 + geom_abline(slope = 1, intercept = 0)
  g2 <- g2 + scale_color_manual(values=c("#000000","#FF0000"))
  g2 <- g2 + xlab("Observed") + ylab("Predicted") + ggtitle('')
  g2 <- g2 + facet_wrap(~Model,
                        scales="free",
                        labeller = labeller(Model = Names))
  g2 <- g2 + theme_bw(base_size = 18)
  g2 <- g2 + theme(legend.position = "bottom",
                   legend.title = element_blank(),
                   axis.text=element_text(size=18),
                   legend.text=element_text(size=rel(1)),
                   plot.title = element_text(hjust=0.5),
                   text=element_text(family="Times New Roman"),
                   axis.title=element_text(size=18))
  print(g2)
}
