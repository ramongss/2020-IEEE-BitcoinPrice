bp_error <- function(data){
  levels(data$Forecast) <- c('One','Two','Three')
  levels(data$Models) <- c('(A)','(B)','(C)','(D)','(E)','(F)','(G)',
                           '(H)','(I)','(J)','(K)','(L)','(M)','(N)')
  
  p <- ggplot(data, aes(x = Forecast, y = APE)) + xlab('Forecasting Horizon (Days)') + ylab('Absolute Percentual Error (APE)')
  p <- p + geom_violin(width = 0.6, trim = FALSE)
  p <- p + geom_boxplot(width = 0.2) + theme_bw()
  p <- p + stat_summary(fun.y=mean, geom="point", shape=20, size=2, color="black", fill="black")
  p <- p + facet_wrap(~ Models, ncol = 7)
  # p <- p +  theme(strip.background = element_blank(),
  # strip.text.x = element_blank())
  p <- p + scale_y_continuous(labels = scales::percent_format(accuracy = 0.1),
                              limits = c(min(0),max(1)))
  print(p)
}
