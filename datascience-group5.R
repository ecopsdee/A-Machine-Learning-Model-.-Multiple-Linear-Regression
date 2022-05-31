# Decision Tree

# Installing neccessary libraries/packages
# install.packages('caTools')
# install.packages('dplyr')
# install.packages('tidyverse')
# install.packages('dummies')
# install.packages('mlbench')
# install.packages('caret')
# install.packages('rpart.plot')
# install.packages('ggcorrplot')
# install.packages("forecast")
# install.packages("neuralnet")
library(neuralnet)
library(forecast)
library(caTools)
library(dplyr)
library(tidyverse)
library(dummies)   # creates dummy variable
library(mlbench)
library(caret)
library(ggplot2)

# Data Preprocessing

# Importing the dataset
dataset <- read.csv('assign5.csv')

# View dataset
view(dataset)

# View dataset dimension
dim(dataset)

# Remove unwanted columns - S/N, New_price
dataset <- dataset[c(2:12, 14)]

# Check if there are some missing values
summary(is.na(dataset))   

# We have 42 missing values for column seats
# Remove the na in the dataset
dataset <- na.omit(dataset)

# Check if there are some still missing values
summary(is.na(dataset)) # now removed with 5977 rows left

# Create Dummy Variables: Fuel_Type, Transmission and Owner_Type

# Create dummy variable for Fuel_type and name it X
# Fuel_type exist on column 5
X <- c(dataset[,5])
unique(X) # View different values in X
dummy1 <- as.data.frame(model.matrix(~ X -1))   # convert to dummy variable
# Replace Fuel_Type Variables with Dummy Values
dataset$Fuel_Type <- dummy1

# Create dummy variable for Transmission and name it T
# Transmission exist on column 6
T <- c(dataset[,6])
unique(T) # View different values in T
dummy2 <- ifelse(T == "Manual", 1, 0)   # convert to dummy variable
# Replace Transmission Variables with Dummy Values
dataset$Transmission <- dummy2

# Create dummy variable for Owner_Type and name it O
# Owner_Type exist on column 7
O <- c(dataset[,7])
unique(O) # View different values in T
dummy3 <- as.data.frame(model.matrix(~ O -1))   # convert to dummy variable
# Replace Owner_Type Variables with Dummy Values
dataset$Owner_Type <- dummy3


## Convert Engine, Power, Mileage, Location and Car to Numeric
# Engine
for (x in 1:nrow(dataset)) {
  z=substring(dataset$Engine[x],-1, 4)
  dataset$Engine[x] <- as.numeric(z)
}
dataset$Engine <- as.numeric(dataset$Engine)

# Power
for (x in 1:nrow(dataset)) {
  dataset$Power[x] <- str_replace_all(dataset$Power[x],"[a-z]"," ")
}
dataset$Power <- as.numeric(dataset$Power)

# Mileage
for (x in 1:nrow(dataset)) {
  dataset$Mileage[x] = str_replace_all(dataset$Mileage[x], ("[a-z, /]")," ")
}
dataset$Mileage <- as.numeric(dataset$Mileage)

# Location
dataset$Location <- as.numeric(as.factor(dataset$Location))

# Car
dataset$Name <- as.numeric(as.factor(dataset$Name))

# Get Unique values for Location and assign it to location code
unique(dataset$Location)
locationsCode = c(legend="Ahmedabad,Bangalore,Chennai,Coimbatore,Delhi,Hyderabad,Jaipur,Kochi,Kolkata,Mumbai,Pune")


## Feature Selection
#Using Location Kilometers_Driven Mileage Engine and Power to Predict price

# View our dataset after Data Preprocessing
str(dataset)

# Check for missing data
summary(dataset)

# Remove Missing Data
dataset <- na.omit(dataset)

# Assigning features of matrix needed for feature selection
# Location Kilometers_Driven Mileage Engine and Power

# Calculate correlation matrix
correlationMatrix <- cor(dataset[,c(2,4,8:10,12)])

# Summarize correlation matrix
print(correlationMatrix)

# Plot correlation
library(ggcorrplot)
ggcorrplot(correlationMatrix)

# Get Correlated attributes. Always between 0.5 and 0.7
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)

# print indexes of highly correlated attributes
print(highlyCorrelated) # No correlation

# Dataset used for feature selection
dataset_new <- dataset[, c(2, 4, 8:10, 12)] # 8:10 - Mileage Engine and Power
str(dataset_new)

### Splitting the dataset into the Training set and Test set

# Set seed to 7 to ensure that result are repeatable
set.seed(7)
# Splitting
split <- sample(2, nrow(dataset_new), replace = TRUE, prob = c(0.7, 0.3))
training_set <- dataset_new[split==1, ]
test_set <- dataset_new[split==2, ]

# Print training and test set
print(training_set)
print(test_set)

### Normalization to test for accuracy
Price_max_pre_normalize = max(test_set$Price)
Price_min_pre_normalize = min(test_set$Price)

# Min-Max Normalization Formula and initialization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalizing the training dataset and reasigning
training_set$Kilometers_Driven <- normalize(training_set$Kilometers_Driven)
training_set$Mileage <- normalize(training_set$Mileage)
training_set$Engine <- normalize(training_set$Engine)
training_set$Power <- normalize(training_set$Power)
training_set$Price <- normalize(training_set$Price)
str(training_set)
# Normalizing the test dataset and reasigning
test_set$Kilometers_Driven <- normalize(test_set$Kilometers_Driven)
test_set$Mileage <- normalize(test_set$Mileage)
test_set$Engine <- normalize(test_set$Engine)
test_set$Power <- normalize(test_set$Power)
test_set$Price <- normalize(test_set$Price)
str(test_set)


# Multiple Linear Regression

model <- lm(Price ~ Kilometers_Driven +Mileage+Power+Engine, data=training_set)
summary(model)
confint(model)

# Fitting Multiple Linear Regression to the Training set
mlrPrice<-predict(model,test_set,method="anova")
results <- data.frame(actual = test_set$Price, prediction = mlrPrice)

ggplt+geom_smooth(model)
# Accuracy
predicted=results$prediction * (Price_max_pre_normalize - Price_min_pre_normalize) + Price_min_pre_normalize
actual=results$actual * (Price_max_pre_normalize - Price_min_pre_normalize) + Price_min_pre_normalize

residual= actual - predicted
deviation=(sum(diag(actual-predicted))/sum(actual))
final_result <- data.frame(actual,predicted,residual)
head(final_result)
accuracy(actual, predicted)
accuracy=(abs(mean(deviation))) * 100
cat("Accuracy for Multiple Linear Regression: ", accuracy)

# Neural Network

#We can chage the hidden layers configuration to c(2,1)
nn <- neuralnet(Price ~Kilometers_Driven +Mileage+Power+Engine,data=training_set, 
                      hidden=c(2,1), linear.output=TRUE, threshold=0.01)
nn$result.matrix

# plot our neural network
plot(nn, rep = 1)

# Prediction
output <- compute(nn, rep = 1, test_set)
head(output$net.result)
results <- data.frame(actual = test_set$Price, prediction = output$net.result)
head(results)

# Accuracy
predicted=results$prediction * (Price_max_pre_normalize - Price_min_pre_normalize) + Price_min_pre_normalize
actual=results$actual * (Price_max_pre_normalize - Price_min_pre_normalize) + Price_min_pre_normalize

residual= actual - predicted
deviation=(sum(diag(actual-predicted))/sum(actual))
final_result <- data.frame(actual,predicted,residual)
head(final_result)
accuracy(predicted, actual)
accuracy=(abs(mean(deviation))) * 100
cat("Accuracy for Neural Network: ", accuracy)

## Regression Tree

# Load Package
library(rpart)

# Fitting Decision Tree Regression to the dataset
regressor <- rpart(Price ~ Kilometers_Driven+Mileage+Power+Engine,
             method = "anova",data=training_set )

# Visualising the Decision Tree Regression results (higher resolution)
library(rpart.plot)
rpart.plot(regressor)
regressor$variable.importance

# Print regressor
print(regressor)

# Predicting a new result with Decision Tree Regression
cat("Predicted value:\n")

# Predicted Value
predPrice <- predict(regressor, test_set, method = "anova")
results <- data.frame(actual = test_set$Price, prediction = predPrice)
head(results)


# Accuracy
predicted=results$prediction * (Price_max_pre_normalize - Price_min_pre_normalize) + Price_min_pre_normalize
actual=results$actual * (Price_max_pre_normalize - Price_min_pre_normalize) + Price_min_pre_normalize
residual= actual - predicted
deviation=(sum(diag(actual-predicted))/sum(actual))
final_result <- data.frame(actual,predicted,residual)
head(final_result)
accuracy(actual, predicted)
accuracy=(abs(mean(deviation))) * 100
cat("Accuracy for Decision Tree: ", accuracy)





