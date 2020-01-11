PO<-function(date,Obs,Pred)
{
  data <- data.frame(as.vector(unlist(data.frame(Obs,Pred))),
                     rep(c("Observed","Prediction"),each=dim(Obs)[1]),
                     rep(date),
                     rep(c(1:dim(Obs)[1]),times=2))
  colnames(data) <- c("Predictions","Legend","Frequency","Times")
  
  g2 <- ggplot(data, aes(Frequency, Predictions,  colour=Legend))+ylab("Fire spots")+xlab("Date")+ggtitle("")
  g2 <- g2 + geom_line(aes(linetype=Legend, colour=Legend), size=1)+ theme_bw(base_size = 22)
  g2 <- g2 + scale_color_manual(values=c("#0000ff","#FF0000"))
  g2 <- g2 + annotate("text", x = data$Frequency[dim(dataset)[1]*0.2], y=max(Obs), label = "Training set", size=6)
  g2 <- g2 + annotate("text", x = data$Frequency[dim(dataset)[1]*0.8], y=max(Obs), label = "Test set", size=6)
  g2 <- g2 + geom_vline(aes(xintercept=as.numeric(Frequency[dim(dataset)[1]*0.7])),size = 1, color ="black")
  # g2 <- g2 + scale_color_manual(values=c("#0000ff","#FF0000"),labels = c("Observed","STL-Ensemble Model"))
  g2 <- g2 + theme(legend.position = "bottom", legend.direction = "horizontal",
                   plot.title = element_text(hjust = 0.5),
                   legend.title = element_blank(),
                   legend.text = element_text(size=20),
                   axis.text=element_text(size=20)) 
  print(g2)
}

# "#0000ff","#FF0000"

# ggplot(Long, aes(x = Date, y = Cases)) + 
#   geom_line(aes(linetype=Models,colour=Models),size=1) +
#   scale_color_manual(values=c("#FF0000","#000000")) +
#   xlab("Date") +  ylab("Dengue Cases Number")+
#   theme_bw(base_size = 18)+
#   theme(legend.position = "bottom", 
#         legend.direction = "horizontal",
#         legend.title = element_blank(),
#         #legend.position = c(0.1, 0.9),
#         legend.background = element_rect(fill="transparent",colour=NA),
#         legend.text = element_text(size=20),
#         axis.text=element_text(size=20),
#         #text=element_text(family="Times New Roman"),
#         axis.title=element_text(size=20))+
#   geom_vline(aes(xintercept=as.numeric(Date[92])))+
#   annotate("text", x = Long$Date[48], y = max(round(Long[,1],2)), label = "Training ",size=6)+
#   annotate("text", x = Long$Date[108], y = max(round(Long[,1],2)), label = "Test",size=6)+
#   scale_x_date(date_labels = "%Y")