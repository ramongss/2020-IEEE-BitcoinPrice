PO1<-function(date,Obs,Pred1,Pred2,Pred3,Pred4,Pred5,Pred6,Pred7
              ,nameM1,nameM2,nameM3,nameM4,nameM5,nameM6,nameM7)
{
  data1 <- data.frame(as.vector(unlist(data.frame(Obs,Pred1,Pred2,Pred3,Pred4,
                                                  Pred5,Pred6,Pred7))),
                   rep(c("Obs","P1","P2","P3","P4","P5","P6","P7"),each=dim(Obs)[1]),
                   rep(date,times=8))
  colnames(data1)<-c("Predictions","Legend","Frequency")
  
  g2 <- ggplot(data1, aes(Frequency, Predictions, colour=Legend))+ylab("")+xlab("")
  g2 <- g2 + geom_line(size=0.8)+ theme_bw(base_size = 22)
  g2 <- g2 + geom_text(x=dim(Obs)[1]*0.1, y=max(Obs),label="Training set",show.legend = FALSE, size = 6, color ="black")
  g2 <- g2 + geom_text(x=dim(Obs)[1]*0.8, y=max(Obs),label="Testing set",show.legend = FALSE, size = 6, color ="black")
  g2 <- g2 + geom_vline(xintercept = dim(Obs)[1]*0.7, size = 1, color ="black")
  g2 <- g2 + scale_color_brewer(palette = "Set1",label = c("Observed",nameM1,nameM2,nameM3,nameM4,nameM5,nameM6,nameM7))
  g2 <- g2 + theme(legend.position = "bottom", 
                   legend.direction = "horizontal",
                   plot.title = element_text(hjust = 0.5),
                   axis.text=element_text(size=18),
                   legend.title = element_blank(),
                   axis.title=element_text(size=20))
  print(g2)
}
