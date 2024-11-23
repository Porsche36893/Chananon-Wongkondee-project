#Importing File and Library

library(zoo)
library(fpp3)
library(forecast)
library(tseries)
library(padr) 
library(zoo)
library(imputeTS)
library(ggplot2)
library(imputeTS) 
fulldata = read.csv("C:/Users/Ppeachary/Downloads/veg.csv")
View(fulldata)

# Monthly Transformation
monthlyfulldata <- aggregate(
  fulldata[, !names(fulldata) %in% "Date"], # Exclude Date column from aggregation
  by = list(MonthYear = as.yearmon(fulldata$Date)), # Group by Year-Month
  FUN = min
)
monthlyfulldata$MonthYear <- as.POSIXct(as.yearmon(monthlyfulldata$MonthYear), format = "%Y-%m")
View(monthlyfulldata)

class(monthlyfulldata$MonthYear)

#------------------------------------EXPLORATORY DATA ANALYSIS----------------------------------------

#1. Plotting Time Series of Okara
ggplot(monthlyfulldata,aes(x = MonthYear,y = Minimum)) + 
  geom_line(color = "darkgreen") +
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  ggtitle("Original Minimum Okara Price Over Time")


#Checking Distribution
ggplot(monthlyfulldata, aes(x = Minimum)) +
  geom_histogram(fill = "darkgreen", color = "black") +
  labs(x = "Minimun Price", y = "Frequency") +
  ggtitle("Histogram of Data")


#2. Time-Series Decomposition
monthlyfulldata.ts = ts(monthlyfulldata$Minimum, frequency = 12)
monthlyfulldata_decompose = stl(monthlyfulldata.ts,s.window = 12)
plot(monthlyfulldata_decompose)

#Trend
adf.test(monthlyfulldata.ts)

ggplot(monthlyfulldata,aes(x = MonthYear,y = Minimum)) + 
  geom_line(color = "darkgreen") +
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  geom_smooth(method = "lm", se = FALSE,color = "navy") +  # Add regression line
  ggtitle("Minimum Okara Price Over Time with Regression Line")
View(monthlyfulldata.ts)
#Seasonality

#Plotting with Zoom in Data
monthlyfulldata_zoom = monthlyfulldata %>% 
  filter(MonthYear >= "2018-01-01 07:00:00
", MonthYear <= "2020-01-01 07:00:00
")


ggplot(monthlyfulldata_zoom, aes(x = MonthYear, y = Minimum)) + 
  geom_line(color = "darkgreen", size = 0.75, alpha = 0.6) +
  theme(panel.grid.major = element_line(color = "burlywood", size = 0.3),
        panel.grid.minor.x = element_line(color = "burlywood", size = 0.3)) +
  geom_smooth(method = "loess", se = FALSE, span = 0.5, color = "navy") +
  ggtitle("Minimum Okara Price from 2018 to 2020 with Moving Average Smoothing")


#3. Correlation plot
acf(monthlyfulldata$Minimum)
pacf(monthlyfulldata$Minimum)

#Independence Testing
Box.test(monthlyfulldata.ts)

#4. Detecting Outliers
outliers = data.frame(tsoutliers(monthlyfulldata.ts))
print(outliers)
nrow(outliers)
#------------------------------PREPROCESSING FOR MONTHLY DATA---------------------------------------


#DATA PREPROCESSING

#1.Missing Value Observation 
View(monthlyfulldata)
is.regular(monthlyfulldata.ts)
is.na(monthlyfulldata.ts)

#2. Splitting Training set And Test Set
#note : split point is October 14, 2019 
train_set = monthlyfulldata %>% 
  filter(MonthYear >= "2013-06-01 07:00:00", MonthYear <= "2019-10-01 07:00:00")

test_set = monthlyfulldata %>% 
  filter(MonthYear >= "2019-11-01 07:00:00") 

summary(train_set)
summary(test_set)

#3. Outliers Detection and Replacement
#Detecting outlier on training set
outliers = tsoutliers(train.ts)
print(outliers)
View(train_set)
View(test_set)

#--------------------------------------SARIMA-----------------------------------------
# Augmented Dickey-Fuller Test and KPSS Test
adf_test <- adf.test(train_ts, alternative = "stationary")
print(adf_test)
kpss_test <- kpss.test(train_ts)
print(kpss_test)

# Plot ACF and PACF
acf(train_set$Minimum, lag.max=100)
pacf(train_set$Minimum, lag.max=100)

#From the graph do the 1st order diff
diff_train_ts <- diff(train_ts, differences = 1)

# Plot ACF and PACF of differenced series
acf(diff_train_ts, main = "ACF of Differenced Train Series")
pacf(diff_train_ts, main = "PACF of Differenced Train Series")


#Final Model
library(astsa)
model_final <- sarima(train_ts, 3, 1, 3, 0, 0, 3, 12, no.constant = TRUE)

#Check ljung-box test
residuals_model <- model_final$residuals
Box.test(residuals_model, lag = 20, type = "Ljung-Box")

#plot forecast graph
sarima_forecast <- sarima.for(train_ts, n.ahead = 12, p = 3, d = 1, q = 3, P = 0, D = 0, Q = 3, S = 12)

# Extract the forecasted values
forecast_values <- as.numeric(sarima_forecast$pred)
test_ts <- as.numeric(test_ts)
# Calculate errors
mae <- mean(abs(forecast_values - test_ts))
rmse <- sqrt(mean((forecast_values - test_ts)^2))
mape <- mean(abs((forecast_values - test_ts) / test_ts)) * 100

# Print the results
cat("MAE:", mae, "\n")
cat("RMSE:", rmse, "\n")
cat("MAPE:", mape, "%\n")

#------------------------------------ DECOMPOSING --------------------------------------------
# Decomposition with STL
monthlytrain_decompose <- stl(train_ts, s.window = "periodic")
plot(monthlytrain_decompose)
#--------------------------------------------ETS----------------------------------------------------
# Convert training set to time series
train_ts <- ts(train_set$Minimum, start = c(2013, 6), frequency = 12)
test_ts <- ts(test_set$Minimum, start = c(2019, 11), frequency = 12)

# Fit a manually specified ETS model
manual_ets_model1 <- ets(train_ts, model = "MNM")
summary(manual_ets_model1)


manual_ets_model2 <- ets(train_ts, model = "MNA")
summary(manual_ets_model2)

manual_ets_model3 <- ets(train_ts, model = "ANA")
summary(manual_ets_model3)

manual_ets_model4 <- ets(train_ts, model = "AAA")
summary(manual_ets_model4)

# Forecast using the ETS model
forecast_ets3 <- forecast(manual_ets_model3, h = 19)



autoplot(forecast_ets3) +
  ggtitle("ETS (A, N, A) Forecast") +
  xlab("Year") +
  ylab("Minimum Price")

#------------------------------------ MODEL EVALUATION --------------------------------------------
# Evaluate forecast accuracy
forecast_values3 <- as.numeric(forecast_ets3$mean)
mae3 <- mean(abs(forecast_values3 - test_ts))
rmse3 <- sqrt(mean((forecast_values3 - test_ts)^2))
mape3 <- mean(abs((forecast_values3 - test_ts) / test_ts)) * 100

# Print evaluation metrics
cat("MAE:", mae3, "\n")
cat("RMSE:", rmse3, "\n")
cat("MAPE:", mape3, "%\n")

