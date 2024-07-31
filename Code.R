#DATA PREPROCESSING

#Importing Data. 
#NOTE: To read the .csv file you you must first download the creditcard.csv file to your working directory
data=read.csv('creditcard.csv')  #Importing the creditcard.csv dataset

#Checking the different characterstics of the data set
str(data)

#Checking For Missing Data

sum(is.na(data))

#There is no missing data in our dataset 

data=data[,-1];data    #Removing the time column since it is of no use for us

#Data Manupulation
data$Amount=scale(data$Amount)      # Normalising The Amount (Feature Scaling)
head(data,n=20L)      #Checking the first 20 lines of the edited data

#Data Visualization
table(data$Class)
barplot(table(data$Class),col=c('yellow','red'),main='Barplot For Understatnding Fraud Detection',
            xlab='Detected-Nondetected',ylab='Number Of Users')    
legend('topright',legend=c('Non-detected(284315)','Detected(492)'),fill=c('yellow','red'))    
                  #getting the idea of how many cases of fraud has been detected graphically as well as with tabulal representation

#Splitting the dataset into training and testing data
 
#You need to install caTools library in your R
install.packages('caTools')
library(caTools)
#We will not be splitting the data set in a particular way.It will be random. So set the seeds to get same results.
set.seed(123456789)

split=sample.split(data$Amount,SplitRatio=0.8)
training_set=subset(data,split==TRUE)
testing_set=subset(data,split==FALSE)


#MODEL BUILDING

#Fitting logistic model on training_set
logit_model=glm(Class~.,data= training_set, family = binomial(link = 'logit'))
summary(logit_model)

#Predicting the Class using our model in the form of probability
predict_class=predict(logit_model,type = 'response',newdata = training_set[,-30])    #We're deleting the last column in order to choose only the predictors
pred=as.vector(predict_class)

#Choosing the cutpoint

#Checking Different approaches for an perfect cut point

#1. using Youden's J statistic

# Load necessary libraries
install.packages("pROC")
library(pROC)


# Generate ROC curve
roc_obj <- roc(training_set$Class , pred)

# Plot the ROC curve
plot(roc_obj)

# Calculate the optimal cutoff using Youden's J statistic
optimal_cutoff <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
optimal_cutoff=optimal_cutoff$threshold

# Print the optimal cutoff
print(optimal_cutoff)

# Classify probabilities into 0 or 1 using the optimal cutoff
predicted_classes_optimal <- ifelse(pred > optimal_cutoff, 1, 0)

# Print the first few predicted classes using the optimal cutoff
head(predicted_classes_optimal)
table(predicted_classes_optimal)
table(training_set$Class)

#2. Using general 0.5 as a cutpoint.
predicted_classes_optimal1 <- ifelse(pred > 0.5, 1, 0)
table(predicted_classes_optimal1)
table(training_set$Class)


#Observe that 0.5 as cutpoint estimates the number of credit card fraud more accurately. So we choose it as our cutpoint.


#Checking our model on testing_set
predict_class_test=predict(logit_model,type = 'response',newdata = testing_set[,-30])    #We're deleting the last column in order to choose only the predictors
pred_test=as.vector(predict_class_test)
predicted_classes_optimal_test <- ifelse(pred_test > 0.5, 1, 0)
table(predicted_classes_optimal_test)
table(testing_set$Class)

#ROC curve
roc_obj_test=roc(testing_set$Class,pred_test)
roc_obj_test
roc_curve=plot(roc_obj_test,main='ROC Curve for The Testing Set',col='red');roc_curve
plot(logit_model)
plot(sort(pred_test),xlim = c(52750,52900),col='blue',xlab = 'Transaction',ylab = 'Proability of Being Fraud Transaction',main = 'Plot to Fnderstand Fraud Transaction Probabily vs Transaction',type = 'l')
#GOODNESS OF FIT
Y=testing_set$Class
Y_hat=predicted_classes_optimal_test
data.frame(Y,Y_hat)
n=length(pred_test)
MAD=100*(1/n)*sum(abs(Y-Y_hat));MAD      #Mean Absolute Deviation
auc_value=auc(roc_curve);auc_value*100  #Area under ROC curve
Residual_plot=plot(residuals(logit_model),col='purple',xlab = 'Transaction',ylab = 'Residuals',main = 'Residual Plot for the Logistic Model in Testing Set')       #Residual Plot
Residual_plot
table(Y,Y_hat)
#Misclassification Error
df=data.frame(Y,Y_hat)
ME=1-sum(df[,1]==df[,2])/length(Y)
ME*100

#SIGNIFICANT PREDICTORS USING TRAINING DATA SET

summary(logit_model)
logit_model_sign=glm(Class~I(V4)+I(V8)+I(V9)+I(V10)+I(V13)+I(V14)+I(V20)+I(V21)+I(V22)+I(V27)+I(V28)+I(Amount),data= training_set, family = binomial(link = 'logit'))
summary(logit_model_sign)
#Predicting the Class using our significant model in the form of probability
predict_class_sign=predict(logit_model_sign,type = 'response',newdata = training_set[,-30])    #We're deleting the last column in order to choose only the predictors
pred=as.vector(predict_class_sign)

#Checking our model on testing_set for signficant predictor model
predict_class_test_sign=predict(logit_model_sign,type = 'response',newdata = testing_set[,-30])    #We're deleting the last column in order to choose only the predictors
pred_test_sign=as.vector(predict_class_test)
predicted_classes_optimal_test_sign <- ifelse(pred_test_sign > 0.5, 1, 0)
table(predicted_classes_optimal_test_sign)


#ROC and other curves using significant predictors
roc_obj_test_sign=roc(testing_set$Class,pred_test_sign)
roc_obj_test_sign
roc_curve=plot(roc_obj_test_sign,main='ROC Curve Using With Significant Predictors In Testing Set',col='red');roc_curve
plot(sort(pred_test_sign),xlim = c(52750,52900),col='blue',xlab = 'Transaction',ylab = 'Proability of Being Fraud Transaction',main = 'Plot to Understand Fraud Transaction Probabily vs Transaction With Significant Predictors',type = 'l')

#Goodness of Fit
Y_hat1=predicted_classes_optimal_test_sign
#Misclassification Error
df=data.frame(Y,Y_hat1)
ME_sign=1-sum(df[,1]==df[,2])/length(Y)
ME_sign*100

auc_value_sign=auc(roc_curve);auc_value_sign*100    #Area under ROC curve





