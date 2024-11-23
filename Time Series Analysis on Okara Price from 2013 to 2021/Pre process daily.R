#Installing Important Library
library(fpp3)
library(forecast)
library(tseries)
library(padr) 
library(zoo)
library(imputeTS)
#Reading/Importing data
fulldata = read.csv("C:/Users/Ppeachary/Downloads/veg.csv")

#Formatting and Selecting Relevent Features
fulldata$Date <- as.Date(fulldata$Date, format = "%m-%d-%y")

#Selecting Okara
veg = fulldata[fulldata$Commodity == "Okara", ]
veg = veg %>%
  arrange(Date)
#-------------------------------------EXPLORATORY DATA ANALYSIS----------------------------------------

#1. Plotting Time Series of Okara
ggplot(veg,aes(x = Date,y = Minimum)) + 
  geom_line(color = "darkgreen",size = 0.8) +
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3)) +
  scale_x_date(date_labels = "%Y",date_breaks = "1 year") +
  ggtitle("Original Minimum Okara Price Over Time")

#Checking Distribution
ggplot(veg, aes(x = Minimum)) +
  geom_histogram(fill = "darkgreen", color = "black") +
  labs(x = "Minimun Price", y = "Frequency") +
  ggtitle("Histogram of Data")

#2. Time-Series Decomposition
veg.ts = ts(veg$Minimum, frequency = 365)
veg_decompose = stl((veg.ts),s.window = 365)
plot(veg_decompose)

#Trend and Stationary Checking
adf.test(veg.ts)

ggplot(veg,aes(x = Date,y = Minimum)) + 
  geom_line(color = "darkgreen") +
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  geom_smooth(method = "lm", se = FALSE,color = "navy") +  # Add regression line
  ggtitle("Minimum Okara Price Over Time with Regression Line")

#Seasonality

#Plotting Zoom in Okara Data
veg_zoom = veg %>% 
  filter(Date >= "2018-01-01", Date <= "2020-01-30")

ggplot(veg_zoom,aes(x = Date,y = Minimum)) + 
  geom_line(color = "darkgreen",size= 0.75,alpha = 0.6) +
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  scale_x_date(date_labels = "%b %y",date_breaks = "2 months") +
  geom_smooth(method = "loess", se = FALSE, span = 0.4,color ="navy") +
  ggtitle("Minimum Okara Price from 2018 to 2020 with Loess Smoothing")

#3. Correlation Plot
acf(veg$Minimum, lag.max = 365)
pacf(veg$Minimum)

#Independence Testing
Box.test(veg.ts)
#
#4. Detecting Outliers

outliers = data.frame(tsoutliers(veg.ts))
print(outliers)
nrow(outliers)
#-------------------------------------------PREPROCESSING-----------------------------------

#DATA PREPROCESSING
#1.Missing Value Observation 

#Padding Missing Date
veg_pad = pad(veg, interval = "day")
veg_pad$Commodity = "Okara"
missing_data = veg_pad %>%
  filter(rowSums(is.na(.)) > 0)
missing_data$Minimum = -2


#Observing Missing Value
ggplot(veg_pad,aes(x = Date,y = Minimum)) + 
  geom_line(color = "darkgreen",size = 0.8) +
  geom_point(data = missing_data, aes(x = Date, y = Minimum), color = "red",size =2.5)+
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  ggtitle("Minimum Okara Price Over Time with Missing Value")

#Distribution of Missing Value
temp1 = veg_pad
temp1$Month <- format(temp1$Date, "%Y-%b")
missing_counts <- table(temp1$Month[is.na(temp1$Minimum)])
barplot(missing_counts, main = "Counts of Missing Values by Month and Year", 
        xlab = "Month", ylab = "Count of Missing Values", col = "pink")

#2. Splitting Training Set and Test Set
train_set = veg_pad %>% 
  filter(Date >= "2013-06-16", Date <= "2019-10-06")

test_set = veg_pad %>% 
  filter(Date > "2019-10-06")  
#note : split point is October 6, 2019 

#3. Data Transformation

#Imputing Missing Value

#Creating Function for Kalman Imputation Specific for this Dataset
impute_ts_kalman <- function(df){
  ts = ts(df[,3])
  ts_imputed = na_kalman(ts,model = "StructTS",smooth = TRUE)
  temp = df
  df_imputed = cbind(temp,data.frame(ts_imputed))
  df_imputed = df_imputed[,-3]
  names(df_imputed)[3] <- "Imputed_Minimum"
  return(df_imputed)
}

#Applying Imputation Function
train_imputed = impute_ts_kalman(train_set)
test_imputed = impute_ts_kalman(test_set) 

#Plotting with Imputed Values
trainjoin_df = inner_join(train_imputed, missing_data, by = "Date")
testjoin_df = inner_join(test_imputed, missing_data, by = "Date")

#Plot on Train Set
ggplot(train_set,aes(x = Date,y = Minimum)) + 
  geom_line(color = "darkgreen",alpha = 10,size =0.8) +
  geom_point(data = trainjoin_df, aes(x = Date,y=Imputed_Minimum ),color = "red",size= 1.7)+
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  ggtitle("Minimum Okara Price of Training Set with Imputed Value") 

#Plot on Test Set
ggplot(test_set,aes(x = Date,y = Minimum)) + 
  geom_line(color = "darkgreen",alpha = 10,size = 0.8) +
  geom_point(data = testjoin_df, aes(x = Date,y=Imputed_Minimum ),color = "red",size= 1.7)+
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  ggtitle("Minimum Okara Price of Test Set with Imputed Value")  

#4.Outliers Detection and Replacement

#Detecting Outliers on Training Set
outliers = tsoutliers(ts(train_imputed$Imputed_Minimum))
print(outliers)
print(train_imputed[c(571, 592), ])

#Replacing Outliers with NA
train.ts = ts(train_imputed$Imputed_Minimum)
train.ts[outliers$index] <- NA
temptrain = cbind(train_imputed,data.frame(train.ts))
temptrain = temptrain[,-3]
names(temptrain)[3] <- "Imputed_Minimum"

#Replacing Outliers Using Kalman Filter
train_process = impute_ts_kalman(temptrain)
print(train_process[c(571, 592), ])

#Plotting Outliers Replacements
outlier_data = temptrain %>%
  filter(rowSums(is.na(.)) > 0)
trainjoin_df2 = inner_join(train_process, outlier_data, by = "Date")

ggplot(temptrain,aes(x = Date,y = Imputed_Minimum)) + 
  geom_line(color = "darkgreen",alpha = 10,size = 0.75) +
  geom_point(data = trainjoin_df2, aes(x = Date,y=Imputed_Minimum.x ),color = "red",size= 3)+
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  ggtitle("Minimum Okara Price of Training Set with Outlier Replacement Using Seasonal Kalman Filter")  

#5. Final Training Set and Test set
train_final = train_process
test_final = test_imputed

#---------------------------------------------EXPORTING---------------------------------------------

#Exporting csv File

write.csv(train_final, "train_final.csv", row.names = FALSE)
write.csv(test_final, "test_final.csv", row.names = FALSE)

