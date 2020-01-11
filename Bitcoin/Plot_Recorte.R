Plot_Recorte<-function(date,Obs,Pred1,Pred2,Pred3) {
  Obs  <- as.data.frame(Obs[1626:1715])
  Pred1 <- as.data.frame(Pred1[1626:1715])
  Pred2 <- as.data.frame(Pred2[1626:1715])
  Pred3 <- as.data.frame(Pred3[1626:1715])
  date <- as.data.frame(as.Date(date[1626:1715],'%m/%d/%Y'))
  n <- dim(Obs)[1]
  cut <- 0.7*n
  
  data <- data.frame(as.vector(unlist(data.frame(Obs,Pred1,Pred2,Pred3))),
                     rep(c("Observed","One-day-ahead\nVMD-STACK-BOXCOX",
                            "Two-days-ahead\nVMD-STACK-CORR",
                            "Three-days-ahead\nVMD-STACK-CORR"),
                         each=dim(Obs)[1]),
                     rep(date),
                     rep(c(1:dim(Obs)[1]),times=4))
  colnames(data) <- c("Predictions","Legend","Date","Times")
  
  data$Legend <- factor(data$Legend, levels = c("Observed","One-day-ahead\nVMD-STACK-BOXCOX",
                                                 "Two-days-ahead\nVMD-STACK-CORR",
                                                 "Three-days-ahead\nVMD-STACK-CORR"))
  
  g2 <- ggplot(data, aes(Date, Predictions,  colour=Legend))+ylab("Bitcoin Price (US$)")+xlab("Day")+ggtitle("")
  g2 <- g2 + geom_line(aes(linetype=Legend, colour=Legend), size=1)+ theme_bw(base_size = 22)
  g2 <- g2 + scale_color_manual(values=c("#000000","#0000ff","#FF0000","#00FF00"))
  g2 <- g2 + scale_x_date(breaks = waiver(),
                          date_labels = "%b %d %Y",
                          date_breaks = "15 days",
                          limits = as.Date(c(date[1,], max = date[dim(date)[1],])))
  # g2 <- g2 + theme(legend.position = "bottom", legend.direction = "horizontal",
  #                  plot.title = element_text(hjust = 0.5),
  #                  legend.title = element_blank(),
  #                  legend.text = element_text(size=18),
  #                  axis.text=element_text(size=18)) 
  g2 <- g2 + theme(legend.direction = "horizontal",
                   plot.title = element_text(hjust = 0.5),
                   legend.justification=c(0,1), 
                   legend.position=c(0.05, 0.99),
                   legend.background = element_blank(),
                   legend.key = element_blank(),
                   legend.title = element_blank(),
                   legend.text = element_text(size=18),
                   axis.text=element_text(size=18))
  print(g2)
}
