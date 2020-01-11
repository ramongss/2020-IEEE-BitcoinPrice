plot_IMF <- function(date,Obs,IMF1,IMF2,IMF3,IMF4,IMF5){
  date    <- as.data.frame(as.Date(date,'%m/%d/%Y'))
  Obs <- as.data.frame(Obs)
  IMF1    <- as.data.frame(IMF1)
  IMF2    <- as.data.frame(IMF2)
  IMF3    <- as.data.frame(IMF3)
  IMF4    <- as.data.frame(IMF4)
  IMF5    <- as.data.frame(IMF5)
  
  legends <- c("Data","IMF[1]","IMF[2]","IMF[3]","IMF[4]","IMF[5]")
  
  data <- data.frame(rep(date),
                     as.vector(unlist(data.frame(Obs,IMF1,IMF2,
                                                 IMF3,IMF4,IMF5))),
                     rep(legends, 
                         each=dim(Obs)[1]),
                     rep(seq(1,dim(Obs)[1])))
  colnames(data) <- c('Date','Observed','Legend','Frequency')
  
  levels(data$Legend) <- legends
  # data$Legend <- factor(data$Legend, levels = legends)
  
  p <- ggplot(data, aes(x=Date, y=Observed)) + ylab("Bitcoin Price (US$)") + xlab("Day")+ggtitle("")
  p <- p + geom_line(size=1, colour='#0000FF')+ theme_bw(base_size = 22)
  p <- p + facet_wrap(.~Legend, ncol = 1, strip.position="left",labeller = label_parsed)
  p <- p + scale_x_date(breaks = waiver(),
                          date_labels = "%Y",
                          date_breaks = "1 year",
                          limits = as.Date(c(date[1,], max = date[dim(date)[1],])))
  p <- p + theme(legend.direction = "horizontal",
                   plot.title = element_text(hjust = 0.5),
                   legend.justification=c(0,1), 
                   legend.position=c(0.05, 0.99),
                   legend.background = element_blank(),
                   legend.key = element_blank(),
                   legend.title = element_blank(),
                   legend.text = element_text(size=18),
                   axis.text=element_text(size=16))
  print(p)
}


