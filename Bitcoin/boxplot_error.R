boxplot_error <- function(Obs, Pred){
	Pred <- as.data.frame(Pred)
	Obs.se  <- (rep(Obs, times = dim(Pred)[1]/length(Obs)))
	Obs.se  <- as.data.frame(Obs.se)
	models <- c('(A)','(B)','(C)','(D)','(E)','(F)','(G)',
	            '(H)','(I)','(J)','(K)','(L)','(M)','(N)')
	
	data <- data.frame(Obs.se,Pred,abs(Obs.se-Pred)/Obs.se,
	                   rep(models,each = length(Obs)))
	colnames(data) <- c('Observed','Preds1','APE','Models')
	levels(data$Models) <- models
	
	p <- ggplot(data, aes(Models, APE)) + xlab('Models') + ylab('Absolute Percentual Error (APE)')
	p <- p + geom_violin()
	p <- p + geom_boxplot(width = 0.2) + theme_bw()
	p <- p + facet_wrap(~ Models, scales = 'free',ncol = 7)
	p <- p +  theme(strip.background = element_blank(),
	                strip.text.x = element_blank())
	p <- p + scale_y_continuous(labels = scales::percent_format(accuracy = 0.1))
	print(p)
}
