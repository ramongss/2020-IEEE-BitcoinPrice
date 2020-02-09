PLOT_train_test <- function(Obs) {
  Obs  <- as.data.frame(Obs)
  n <- dim(Obs)[1]
  cut <- 0.7*n
  data <- data.frame(as.vector(unlist(Obs)),
                     rep(c("Training","Testing"), 
                         times = c(length(Obs[1:cut,]),
                                   dim(tail(Obs,n-cut))[1])),
                     rep(seq(1,dim(Obs)[1])))
  colnames(data) <- c('Observed','Type','Frequency')
  
  data$Type <- factor(data$Type, levels = c("Training", "Testing"))
  
  g2 <- ggplot(data, aes(Frequency, Observed)) + ylab("Power (KW)") + xlab("Samples (10 minutes)") + ggtitle('')
  g2 <- g2 + geom_line(aes(group=1, colour=Type), size=1) + theme_bw(base_size = 16)
  g2 <- g2 + theme(legend.direction = "vertical",
                   plot.title = element_text(hjust = 0.5),
                   legend.justification=c(1,0), 
                   legend.position=c(0.25, 0.75),  
                   legend.background = element_blank(),
                   legend.key = element_blank(),
                   legend.title = element_blank(),
                   legend.text = element_text(size=16),
                   axis.text=element_text(size=13))
  
  g2 <- g2 + scale_color_brewer(palette="Set1")
  
  print(g2)
}




