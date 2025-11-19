- Comment / Variables / Operators / Data types:
```
#Simple R program:
first_str<- "hello world"
first_str

#Comment
#To write program for printing Hello World

# Variable
var.1 = c(1, 2, 3)
var.2<-c("lotus", "rose")
c(FALSE,1)->var.3
var.1
cat("var1 is", var.1, "\n")
cat("var2 is", var.2, "\n")
cat("var3 is", var.3,"\n")

# Arithmetic operators
a<-c(10,20,30,40)
b<-c(2,2,3,4)
cat("sum=", (a+b), "\n")
cat("difference=", (a-b), "\n")
cat("product=", (a*b), "\n")
cat("quotient=", (a/b),"\n")
cat("remainder=", (a%%b), "\n")
cat("int division=", (a%/%b),"\n")
cat("exponent=", (a^b), "\n")

# Relational operator
a<-c(10,20,30,40)
b<-c(25,2,30,3)
10/13/25, 10:59 AM VedangiGholap-15_CDAfile.ipynb - Colab
https://colab.research.google.com/drive/1DeBsAkRaNxnZUKkX_pPobsU_ZDCu9tc_#printMode=true 1/17
cat(a, "less than", b, (a<b), "\n")
cat(a, "greater than", b, (a>b), "\n")
cat(a, "less than or equal to", b, (a<=b),"\n")
cat(a, "greater than or equal to", b, (a>=b), "\n")
cat(a, "equal", b, (a==b), "\n")
cat(a, "not equal to", b, (a!=b), "\n")

# Assignment operator
#leftward assigment
var.a=c(0,20, TRUE)
var.b<-c(0,20, TRUE)
var.c<<-c(0,20, TRUE)
var.a
var.b
var.c
#rightward assignment
c(1,2, TRUE)->v1
c(1,2, TRUE)->>v2
v1
v2

# Numeric data types
  # ----Numeric:
x = 10.9
x
y = 5
y
class(x)
class(y)
is.integer(x)
is.integer(y)
  # ----Integer:
x=as.integer(3)
x
class(x)
is.integer(x)
y=as.integer(3.23)
y
z=as.integer("7.81")
z
as.integer(TRUE)
as.integer(FALSE)
  # ----Complex:
x = 5 + 4i
x
class(x)
is.complex(x)
y=as.complex(3)
y
  # ----Logical:
x=1;
y = 2
z=x>y
z
class(z)
as.logical(1)
as.logical(0)

# Character Data Type
x="abc"
y=as.character(7.8)
x
y
class(y)

# Sequence operator
x<-seq(2, 3, by = 0.2)
x
x<-seq (1,5,length=3)
x
```
- Vectors:
```
# Vectors
x<-1:7
x
y<-2.5:10.5
y
z<-2.5:4.7
z

# typeof
x<-c(10,2.5, TRUE)
x
typeof(x)
x<-c(1,2.8, TRUE, "apple")
typeof(x)
length(x)

# Sequence operator
x<-seq(2, 3, by = 0.2)
x
x<-seq (1,5,length=3)
x

# Combining vectors
n = c(2, 3, 4)
m = c("a", "b", "c","d")
c(n,m)

# Accessing Vector
s=c("a","b", "c", "d", "e")
s[3]
s[-3]
s[10]

s=c("a","b", "c", "d", "e")
s[c(2, 3)]
s[c(2, 3, 3)]
s[c(2, 1, 3)]
s[2:4]

# Logical vectors
s = c(1, 2, 2, 3, - 5, - 6)
s[c(TRUE, FALSE, FALSE, TRUE, FALSE)]
s[s<0]
s[s>0]

# Character vector as index
v=c("Jack", "joe", "tom")
v
names (v)=c("first", "second", "third")
names (v)
v["second"]
v[c("third", "first", "second")]

# Modifying vectors
x = c(10, 20, 30, 40, 50, 60)
x
x[2] <-90
x
x[x<30]<-5
x
x<-x [1:4]
x

# Deleting Vectors
x = c(10, 20, 30, 40, 50)
x
x<-NULL
x
```
- Experiment 2 - Matrices
```
# Creating Matrices

  # Elements are arranged by row
M <- matrix(c(1:12), nrow=4, byrow=TRUE)
M
  # Elements are arranged by collumn
N <- matrix(c(1:12), nrow=4, byrow=FALSE)
N
  # Define column and row names
rnames = c("r1","r2","r3","r4")
cnames = c("c1","c2","c3")
P <- matrix(c(1:12), nrow=4, byrow=TRUE,
dimnames = list(rnames, cnames))
P

# Creating Matrices using Functions
  # Elements be filled collumn-wise
M = cbind(c(1,2,3), c(4,5,6))
M
  # Elements be filled row-wise
N = rbind(c(1,2,3), c(4,5,6))
N

# Factors
  # Factor Creation
x <- factor(c("single", "married", "married", "single", "divorced"))
x
class(x)
levels(x)
str(x)

# Data Frames
x <- data.frame("roll"=1:2, "name"=c("Jack", "Jill"), "age"=c(20,22))
x
names(x)
nrow(x)
ncol(x)
str(x)
summary(x)
```
- Random sampling: with and without replacement
- Stratified sampling in R with reproducible sample
```
# Expt 3 
sample(1:20,10)

sample (1:6,4, replace=TRUE)

sample(1:6,4, replace=FALSE)

sample(LETTERS)

data<-c(1,3,5,6,7,8,9,10,11,12,14)
sample(x=data, size=5)

data<-c(1,3,5,6,7,8,9,10,11,12,14)
sample(x=data, size=5, replace=TRUE)

df<-data.frame (x = c(3, 4, 5, 6, 8, 12, 14),
y = c(12, 6, 4, 23, 25, 8, 9),
z = c(2, 7, 8, 8, 15, 17, 29) )
df

rand_df<-df [sample(nrow (df),size=3),]
rand_df

install.packages("dplyr")

library(dplyr)

set.seed(1)
df <- data.frame(
  grade=rep(c("Freashmen", "Sophomore", "Junior", "Senior"), each=15),
  gpa = rnorm(60, mean = 85, sd = 3 ) # generates random GPA values
)
head(df)

start_sample<-df%>%
group_by(grade)%>% sample_n(size=10)
table(start_sample$grade)

library(dplyr)
start_sample<-df%>%
group_by(grade)%>%
slice_sample (n = 15)
table(start_sample$grade)
```
- Calculator: measure of central tendency mean median and mode
```
marks<-c(97,67,89,34)
result<-mean(marks)
print(result)

marks<-c(97,67,89,34)
result<-median (marks)
print(result)

marks<-c(97,67,68,89,34)
result<-median(marks)
print(result)

marks<-c(97,67,89,34,97)
mode=function(){
return(names (sort(-table (marks)))[1])
}
mode()

#
data<-data.frame(
Product=c("TM195", "TM195", "TM195", "TM195", "TM195", "TM195"),
Age=c(18,19,19,19,20,20),
Gender=c("Male", "Male", "Female", "Male", "Male", "Female"),
Education=c(14,15,14,12,13,14),
MaritalStatus=c("Single", "Single", "Partnered", "Single", "Partnered", "Partnered"),
Usage=c(3,2,4,3,4,3),
Fitness=c(4,3,4,3,2,3),
Income=c(29562, 31836, 30699, 28465, 75643,61243),
Miles=c(12,75,66,85,47,66)
)
write.csv(data, "data.csv", row.names=FALSE)
cat("CSV file created")

mydata=read.csv("data.csv", stringsAsFactors = F)
print(head(mydata))

mydata=read.csv("data.csv", stringsAsFactors = F)
result=mean(mydata$Age)
print(result)

mydata=read.csv("data.csv", stringsAsFactors = F)
result=median(mydata$Age)
print(result)

mydata=read.csv("data.csv", stringsAsFactors = F)
mode=function(){
return (names (sort(-table(mydata$Age))) [1])
}
mode()
```
- Measure of variability : range, standard deviation, variance, percentile, interquartile range
```
x<-c(5, 6, 7, 3, 12, 44)
print(range(x))
print(max(x-min(x)))

d<-sqrt(var(x))
x<-c(5, 6, 7, 3, 12, 44)
print(d)

x<-c(5, 6, 7, 3, 12, 44)
print(var(x))

x<-c(5, 6, 7, 3, 12, 44)
res<-quantile(x, probs=0.5)
res

x<-c(5, 6, 7, 3, 12, 44)
print(IQR(x))

print(head (mydata))

print(range(mydata$Miles))

print(max(mydata$Miles)-min(mydata$Miles))

print(sqrt(var (mydata$Miles)))

print(sd(mydata$Miles))

print(var(mydata$Miles))

print(quantile (mydata$Miles))

print(IQR(mydata$Miles))
```
- Data vizualization in R
```
# Bar Plot
# Horizontal Bar Plot
barplot(airquality$Ozone,
  main = 'Ozone Concenteration in air',
  xlab = 'ozone levels', horiz = TRUE)
# Vertical Bar Plot
barplot(airquality$Ozone,
  main = 'Ozone Concenteration in air',
  xlab = 'ozone levels', col ='blue', horiz = FALSE)

# Histogram
hist(airquality$Temp, main ="La Guardia Airport's Maximum Temperature(Daily)",
  xlab ="Temperature(Fahrenheit)",
  xlim = c(50, 125), col ="yellow",
  freq = TRUE)

# Box Plot
boxplot(airquality$Wind, main = "Average wind speed at La Guardia Airport",
  xlab = "Miles per hour", ylab = "Wind",
  col = "orange", border = "brown",
  horizontal = TRUE, notch = TRUE)

# Multiple box plots
boxplot(airquality[, 0:4],
  main ='Box Plots for Air Quality Parameters')

# Scatter Plot
plot(airquality$Ozone, airquality$Month,
  main ="Scatterplot Example",
  xlab ="Ozone Concentration in parts per billion",
  ylab =" Month of observation ", pch = 19)

# Heat Map / heatmap
data <- matrix(rnorm(25, 0, 5), nrow = 5, ncol = 5)
colnames(data) <- paste0("col", 1:5)
rownames(data) <- paste0("row", 1:5)
heatmap(data)

# Map
install.packages("maps")
library(maps)
map(database = "world")
df <- data.frame(
  city = c("New York", "Los Angeles", "Chicago", "Houston", "Phoenix"),
  lat = c(40.7128, 34.0522, 41.8781, 29.7604, 33.4484),
  lng = c(-74.0060, -118.2437, -87.6298, -95.3698, -112.0740)
)
points(x = df$lng, y = df$lat, col = "Red")


sqrt(x ^ 2 + y ^ 2)
}

x <- y <- seq(-1, 1, length = 30)
z <- outer(x, y, cone)

persp(x, y, z,
      main="Perspective Plot of a Cone",
      zlab = "Height",
      theta = 30, phi = 15,
      col = "orange", shade = 0.4
      )
```
- Power analysis in Hypothesis Testing
```
from numpy import array
from matplotlib import pyplot as plt
from statsmodels.stats.power import TTestIndPower

effect_sizes = array([0.2, 0.5, 0.8])
sample_sizes = array(range(5, 100))

analysis = TTestIndPower()

# Plot power curves for different effect sizes
fig, ax = plt.subplots()
analysis.plot_power(
  dep_var='nobs', # vary number of observations
  nobs=sample_sizes,
  effect_size=effect_sizes,
  alpha=0.05,
  ax=ax
)

plt.title('Power of Two-Sample t-Test')
plt.xlabel('Sample Size per Group')
plt.ylabel('Statistical Power')
plt.legend(['Small (0.2)', 'Medium (0.5)', 'Large (0.8)'])
plt.grid(True)
plt.show()
```
- Date & Time
```
# Date
# Coerce a 'Date' object from character
x<-as.Date("2004-10-31")
x

# Time
x <- Sys.time()
x

# Get Current Date And Time:

date()

Sys.Date()

Sys.time()

# Get Current Date Using R Lubridate Package:
install.packages('lubridate')

library(lubridate)

now()

# Extraction Years, Months, And Days From Multiple Date Values In R:
library (lubridate)

dates <- c("2025-08-22", "2012-04-19","2017-03-05")

year(dates)
month(dates)
mday(dates)

# Manipulate Date Values In R:
my_date <- as.Date("2022-05-27")
my_date

class(my_date)

format(my_date, "%y-%h-%d")

format(my_date, "%d-%m-%y")

format(my_date, "%d-%m-%Y")

format(my_date, "%Y-%h-%d")

format(my_date, "%Y-%m-%h-%d-%H-%M-%S")

format(my_date, "%d-%m-%y")

# Using Update() To Update Date Values In R:
date <- ymd("2025-08-22")
update(date, year=2004,month=10,mday=12)

update(date, year=2004,month=9, mday=1)

update(date, year=2004,minut=10,seconds=20)
```
- Linear regression in R
```
# Read CSV file
LungCapData <- read.csv("LungCapData.csv", header = TRUE, stringsAsFactors = TRUE)

# View dataset
print(LungCapData)

names(LungCapData)

#
# Simple Linear Regression: Lung Capacity ~ Age
model <- lm(LungCap ~ Age, data = LungCapData)

# Summary of regression model
summary(model)

# Plot the relationship
plot(LungCapData$Age, LungCapData$LungCap,
     main = "Relationship between Age and Lung Capacity",
     xlab = "Age (years)",
     ylab = "Lung Capacity",
     pch = 19, col = "blue")

# Add regression line
abline(model, col = "red", lwd = 2)

#
# Multiple Linear Regression :

# Convert categorical columns to factors
LungCapData$Smoke <- as.factor(LungCapData$Smoke)
LungCapData$Gender <- as.factor(LungCapData$Gender)
LungCapData$Caesarean <- as.factor(LungCapData$Caesarean)

# Multiple Linear Regression: LungCap ~ Age + Height + Smoke + Gender + Caesarean
multi_model <- lm(LungCap ~ Age + Height + Smoke + Gender + Caesarean, data = LungCapData)

# Summary of the model
summary(multi_model)

# Predicted values
pred <- predict(multi_model)

# Plot actual vs predicted
plot(LungCapData$LungCap, pred,
     main = "Actual vs Predicted Lung Capacity",
     xlab = "Actual Lung Capacity",
     ylab = "Predicted Lung Capacity",
     pch = 19, col = "darkgreen")
abline(0, 1, col = "red", lwd = 2)  # 45-degree line for perfect prediction
```
Logistics regression in R
```
# Import Dataset
install.packages("dplyr")
library(dplyr)

head(mtcars)

# Splitting the Dataset
install.packages("caTools") 
library(caTools)

split <- sample.split(mtcars, SplitRatio = 0.8)

train_reg <- subset(mtcars, split == "TRUE")
test_reg <- subset(mtcars, split == "FALSE")

# Building the model
logistic_model <- glm(vs ~ wt + disp,
                    data = train_reg,
                    family = "binomial")
logistic_model

summary(logistic_model)

#
Call: Displays the formula and dataset used for the model.
Deviance Residuals: Show how well the model fits the data. Smaller values indicate a better fit.
Coefficients: Reflect the impact of each predictor on the outcome. Includes standard errors.
Significance Codes: Indicate the statistical significance of each predictor (e.g., ‘***’ means highly significant).
Dispersion Parameter: Set to 1 for logistic regression, as it uses a binomial distribution.
Null Deviance: The model’s deviance without predictors (only the intercept).
Residual Deviance: The model’s deviance after adding predictors. A lower value suggests a better fit.
AIC: Used to compare models. Lower AIC indicates a better model with fewer unnecessary variables.
Fisher Scoring Iterations: The number of steps taken to find the best-fitting model.
#

# Predict test data
predict_reg <- predict(logistic_model,
                       test_reg, type = "response")

predict_reg <- as.data.frame(predict_reg)
predict_reg

# Plotting a Confusion Matrix
library(ggplot2)
library(reshape2)

conf_matrix <- table(test_reg$vs, predict_reg)

# Reshape the confusion matrix for ggplot2
conf_matrix_melted <- as.data.frame(conf_matrix)
colnames(conf_matrix_melted) <- c("Actual", "Predicted", "Count")

ggplot(conf_matrix_melted, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "black", size = 6) +  # Add text labels
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted") +
  theme_minimal()
```
linear regression with gradient descent 
``` Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

learning_rate = 0.1
n_iterations = 100

for _ in range(n_iterations):
y_pred x_b.dot(theta)
gradients (2/m) X_b.T.dot(y_predy)
theta learning_rate gradients

plt.figure(figsize=(10, 5))
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, x_b.dot(theta), color="red", label="Optimized Line (With GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression With Gradient Descent")
plt.legend()
plt.show()
```
