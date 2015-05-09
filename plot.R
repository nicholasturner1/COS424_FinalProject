library(ggplot2)
library(reshape)

setwd('~/Development_Workspaces/COS424_FinalProject/')
#setwd('GitHub/COS424_FinalProject/')

#MSD (Year Prediction Subset)
d <- read.csv('YearPredictionMSD.txt', header=FALSE)

#Adding variable names
mu_names <- paste('mu',seq(1,12),sep='')
sigma_names <- paste('sigma',seq(1,78),sep='')

colnames(d) <- c('year',mu_names,sigma_names)

#Adding year bin features
d$yr5 <- d$year %/% 5 * 5
d$yr10 <- d$year %/% 10 * 10

###############################
#Year Histograms
qplot(d$year, geom='histogram', xlab='Year', ylab='Count', main='Year Distribution')
qplot(d$yr5, geom='histogram', xlab='Year', ylab='Count', main='Year Distribution (by Lustrum)')
qplot(d$yr10, geom='histogram', xlab='Year', ylab='Count', main='Year Distribution (by Decade)')

###############################
#Forming dataset with no songs before 1960 (under-represented)
d_post60 <- d[d$year >= 1960,]
###############################
#Splitting data into training, cv, and test sets
set.seed(987654321)

training_percentage <- 0.8
cv_percentage <- 0.1
#implicitly, test_percentage = 0.1

n <- nrow(d)
n_training <- floor(training_percentage * n)
n_cv <- floor(cv_percentage * n)
n_test <- n - n_training - n_cv

train_index <- sample(seq_len(n), n_training)
train <- d[train_index,]

cv_test <- d[-train_index,]
cv_index <- sample(seq_len(nrow(cv_test)), n_cv)
cv <- cv_test[cv_index,]

test <- cv_test[-cv_index,]

#Writing sets to disk
write.csv(train, 'train.csv',row.names=FALSE)
write.csv(cv, 'cv.csv',row.names=FALSE)
write.csv(test, 'test.csv',row.names=FALSE)
###############################
#Same for post-60 dataset

training_percentage <- 0.8
cv_percentage <- 0.1
#implicitly, test_percentage = 0.1

n <- nrow(d_post60)
n_training <- floor(training_percentage * n)
n_cv <- floor(cv_percentage * n)
n_test <- n - n_training - n_cv

train_index <- sample(seq_len(n), n_training)
train <- d_post60[train_index,]

cv_test <- d_post60[-train_index,]
cv_index <- sample(seq_len(nrow(cv_test)), n_cv)
cv <- cv_test[cv_index,]

test <- cv_test[-cv_index,]

#Writing sets to disk
write.csv(train, 'train_post60.csv',row.names=FALSE)
write.csv(cv, 'cv_post60.csv',row.names=FALSE)
write.csv(test, 'test_post60.csv',row.names=FALSE)
###############################
#Dem plots

#performing SVD
mean_svd <- svd(t(d[2:13]))
qplot(1:length(mean_svd$d),mean_svd$d, geom='point')
#top 3 seem slightly more important than the rest

#collecting first 3 right singular vectors and the year bins into a data frame
to_plot <- data.frame(
  'v1' = mean_svd$v[,1],
  'v2' = mean_svd$v[,2],
  'v3' = mean_svd$v[,3],
  'year' = d$year,
  'yr5' = d$yr5,
  'yr10' = d$yr10)

to_plot$yr10 <- as.factor(to_plot$yr10)
to_plot$yr5 <- as.factor(to_plot$yr5)
#First two comps by decade
ggplot(to_plot, aes(x=v1, y=v2, colour=yr10)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='1st Right Singular Vector Weight',
       y='2nd Right Singular Vector Weight',
       title='V1 x V2, colored by year')

#Same for V1 and V3
ggplot(to_plot, aes(x=v1, y=v3, colour=yr10)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='1st Right Singular Vector Weight',
       y='3rd Right Singular Vector Weight',
       title='V1 x V3, colored by year')

#Same for V2 and V3
ggplot(to_plot, aes(x=v2, y=v3, colour=yr10)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='2nd Right Singular Vector Weight',
       y='3rd Right Singular Vector Weight',
       title='V2 x V3, colored by year')

#All the same for the covariance matrix
cov_svd <- svd(t(d[14:91]))
qplot(1:length(cov_svd$d), cov_svd$d, geom='point')
# #1 is way better than everything else, but I'll include 2 more again

#collecting first 3 right singular vectors and the year bins into a data frame
to_plot2 <- data.frame(
  'v1' = cov_svd$v[,1],
  'v2' = cov_svd$v[,2],
  'v3' = cov_svd$v[,3],
  'year' = d$year,
  'yr5' = d$yr5,
  'yr10' = d$yr10)

to_plot2$yr10 <- as.factor(to_plot$yr10)
to_plot2$yr5 <- as.factor(to_plot$yr5)
#First two comps by decade
ggplot(to_plot2, aes(x=v1, y=v2, colour=yr10)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='1st Right Singular Vector Weight',
       y='2nd Right Singular Vector Weight',
       title='COV V1 x V2, colored by year')

#Same for V1 and V3
ggplot(to_plot2, aes(x=v1, y=v3, colour=yr10)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='1st Right Singular Vector Weight',
       y='3rd Right Singular Vector Weight',
       title='COV V1 x V3, colored by year')

#Same for V2 and V3
ggplot(to_plot2, aes(x=v2, y=v3, colour=yr10)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='2nd Right Singular Vector Weight',
       y='3rd Right Singular Vector Weight',
       title='COV V2 x V3, colored by year')

########
#Error plots

preds <- read.csv('et100_preds.csv', header=FALSE)
cv <- read.csv('cv.csv')

err <- (cv$year - preds)^2
test <- data.frame(
  'year' = cv$year,
  'error' = err)

ggplot(test, aes(x=year, y=V1)) +
  geom_smooth() +
  labs(
    x = "Year",
    y = 'Squared Error',
    title= 'Squared Error by Year: "Yes, Our Data is Skewed We Get It"')

##################
#ExtraTrees Comparison
cv <- read.csv('cv.csv')
p_et10 <- read.csv('preds/ET/ET_10_year_cv_preds_full.csv', header=FALSE)
p_et20 <- read.csv('preds/ET/ET_20_year_cv_preds_full.csv', header=FALSE)
p_et30 <- read.csv('preds/ET/ET_30_year_cv_preds_full.csv', header=FALSE)
p_et40 <- read.csv('preds/ET/ET_40_year_cv_preds_full.csv', header=FALSE)
p_et50 <- read.csv('preds/ET/ET_50_year_cv_preds_full.csv', header=FALSE)

err10 <- as.vector((cv$year - p_et10$V1)^2)
err20 <- as.vector((cv$year - p_et20$V1)^2)
err30 <- as.vector((cv$year - p_et30$V1)^2)
err40 <- as.vector((cv$year - p_et40$V1)^2)
err50 <- as.vector((cv$year - p_et50$V1)^2)

qplot(cv$year, err, geom='smooth') + geom_point(alpha=0.01)

to_plot = data.frame(
  'ET10' = err10,
  'ET20' = err20,
  'ET30' = err30,
  'ET40' = err40,
  'ET50' = err50,  
  'year' = cv$year
  )

m_to_plot <- melt(to_plot, id.vars='year')
colnames(m_to_plot) <- c('Year','Model','Squared_Error')

ggplot(m_to_plot, aes(x=Year, y=Squared_Error, colour=Model, fill=Model)) +
  geom_smooth() +
  geom_point(alpha=0.05) +
  labs(
    y = "Squared Error",
    title = "Squared Error by Year for ExtraTrees Models")

to_plot_p = data.frame(
  'ET10' = p_et10$V1,
  'ET20' = p_et20$V1,
  'ET30' = p_et30$V1,
  'ET40' = p_et40$V1,
  'ET50' = p_et50$V1,
  'year' = cv$year
  )

m_to_plot_p <- melt(to_plot_p, id.vars='year')
colnames(m_to_plot_p) <- c('Year','Model','Predicted')

ggplot(m_to_plot_p, aes(x=Year, y=Predicted, colour=Model, fill=Model)) +
  geom_smooth() +
  geom_abline(intercept=0, slope=1, colour='grey50') +
  labs(
    x = 'Year',
    y = 'Predicted Release Year',
    title = 'ExtraTrees Model Predictions by Year')

###############################
#New SVD (PCA)
d[2:91] <- scale(d[2:91]) #Mean Centering and Scaling Variance to 1
full_svd <- svd(t(d[2:91]))
qplot(1:length(full_svd$d),full_svd$d, geom='point')
#top5 are pretty good, top3 are still better, and top1 better still

#collecting first 3 right singular vectors and the year bins into a data frame
to_plot <- data.frame(
  'v1' = full_svd$v[,1],
  'v2' = full_svd$v[,2],
  'v3' = full_svd$v[,3],
  'Year' = d$year,
  'Lustrum' = d$yr5,
  'Decade' = d$yr10)

to_plot$Decade <- as.factor(to_plot$Decade)
to_plot$Lustrum <- as.factor(to_plot$Lustrum)
#First two comps by decade
ggplot(to_plot, aes(x=v1, y=v2, colour=Decade)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='1st Right Singular Vector Weight',
       y='2nd Right Singular Vector Weight',
       title='V1 x V2, Colored by Decade')

#Same for V1 and V3
ggplot(to_plot, aes(x=v1, y=v3, colour=Decade)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='1st Right Singular Vector Weight',
       y='3rd Right Singular Vector Weight',
       title='V1 x V3, Colored by Decade')

#Same for V2 and V3
ggplot(to_plot, aes(x=v2, y=v3, colour=Decade)) +
  scale_color_brewer(palette='Spectral') +
  geom_point() +
  labs(x='2nd Right Singular Vector Weight',
       y='3rd Right Singular Vector Weight',
       title='V2 x V3, Colored by Decade')

###############################
#LDA Plots
lda_yr <- read.csv('lda_data/train_lda_yr_2comps.csv')
lda_yr5 <- read.csv('lda_data/train_lda_yr5_2comps.csv')
lda_yr10 <- read.csv('lda_data/train_lda_yr10_2comps.csv')

lda_colnames <- c('Year','Component_1','Component_2','Lustrum','Decade')
colnames(lda_yr) <- lda_colnames
colnames(lda_yr5) <- lda_colnames
colnames(lda_yr10) <- lda_colnames

lda_yr$Decade <- as.factor(lda_yr$Decade)

ggplot(lda_yr, aes(x=Component_1, y=Component_2, colour=Decade)) +
  geom_point() +
  scale_color_brewer(palette = "Spectral") +
  labs(
    x = "1st LDA Feature",
    y = "2nd LDA Feature",
    title = "LDA Features (spreading Year)")

lda_yr5$Decade <- as.factor(lda_yr$Decade)

ggplot(lda_yr5, aes(x=Component_1, y=Component_2, colour=Decade)) +
  geom_point() +
  scale_color_brewer(palette = "Spectral") +
  labs(
    x = "1st LDA Feature",
    y = "2nd LDA Feature",
    title = "LDA Features (spreading Lustrum)")

lda_yr10$Decade <- as.factor(lda_yr$Decade)

ggplot(lda_yr10, aes(x=Component_1, y=Component_2, colour=Decade)) +
  geom_point() +
  scale_color_brewer(palette = "Spectral") +
  labs(
    x = "1st LDA Feature",
    y = "2nd LDA Feature",
    title = "LDA Features (spreading Decade)")

###############################
#Melody Error Plots
std_values <- read.csv('Melody_std_values_w_year.csv', header=FALSE)
colnames(std_values) <- c('std_value','year')
std_values$quartile <- (
  (std_values$std_value > quantile(std_values$std_value, 0.2)) +
  (std_values$std_value > quantile(std_values$std_value, 0.4)) +
  (std_values$std_value > quantile(std_values$std_value, 0.6)) +
  (std_values$std_value > quantile(std_values$std_value, 0.8))
  )
std_values$quartile <- as.factor(std_values$quartile)

qplot(std_values$year, 
      std_values$std_value, 
      colour=std_values$quartile, 
      geom="point") +
  scale_color_brewer(palette='Spectral') +
  labs(
    x = 'Year',
    y = 'Standard Deviation between Predictions',
    title = 'Standard Deviation over Predictions by Year',
    colour= 'Split Number')
  
