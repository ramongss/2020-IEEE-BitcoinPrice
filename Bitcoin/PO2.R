PO2<-function(Obs,Pred,Title)
{
  n <- dim(Obs)[1]
  cut <- 0.7*n
  
  data1<-data.frame(Obs,
                    Pred,
                    rep(c("Training","Test"), 
                        times = c(length(Obs[1:cut,]),dim(tail(Obs,n-cut))[1])),
                    rep(seq(1,dim(Obs)[1])))
  colnames(data1)<-c("Observed","Predicted","Type","Frequency")
  
  g2 <- ggplot(data1, aes(x=Observed, y=Predicted))
  g2 <- g2 + geom_point(aes(shape=Type, color = Type),size=3)
  g2 <- g2 + geom_abline(slope = 1, intercept = 0)
  g2 <- g2 + scale_color_manual(values=c("#000000","#FF0000"))
  g2 <- g2 + xlab("Observed") + ylab("Predicted") + ggtitle(Title)
  # g2 <- g2 + facet_wrap(~State,scales="free")
  g2 <- g2 + theme_bw(base_size = 18)
  g2 <- g2 + theme(legend.position = "bottom", 
                   legend.title = element_blank(),
                   axis.text=element_text(size=18),
                   legend.text=element_text(size=rel(1)),
                   plot.title = element_text(hjust=0.5),
                   text=element_text(family="Times New Roman"),
                   axis.title=element_text(size=18))
  print(g2)
  
  # ggsave(
  #   file.path("Plot", paste("predXobs.eps")),
  #   g2,
  #   width = 10,
  #   height = 5,
  #   dpi = 1200)
}
