Traindata = read.csv("C:/Users/Ppeachary/Downloads/train_final.csv")
Testdata = read.csv("C:/Users/Ppeachary/Downloads/test_final.csv")
View(Traindata)
View(Testdata)
summary(Testdata)
Traindata$Date <- as.Date(Traindata$Date, format = "%Y-%m-%d")
Testdata$Date <- as.Date(Testdata$Date, format = "%Y-%m-%d")
View(Testdata)
summary(Testdata)


# Load libraries
library(forecast)
library(tseries)
library(tsibble)

# Convert to time series
train_ts <- ts(Traindata$Imputed_Minimum, frequency = 365)
test_ts <- ts(Testdata$Imputed_Minimum, frequency = 365)


# Augmented Dickey-Fuller Test and KPSS Test
adf_test <- adf.test(train_ts, alternative = "stationary")
print(adf_test)
kpss_test <- kpss.test(train_ts)
print(kpss_test)

# Plot ACF and PACF to determine ARIMA orders
acf(train_ts, main="ACF of Train Series")
pacf(train_ts, main="PACF of Train Series")

#From the graph do the 1st order diff
diff_train_ts <- diff(train_ts, differences = 1)

# Plot ACF and PACF of differenced series
acf(diff_train_ts, main = "ACF of Differenced Train Series")
pacf(diff_train_ts, main = "PACF of Differenced Train Series")
diff_adf_test <- adf.test(diff_train_ts, alternative = "stationary")
print(diff_adf_test)


# Print the best model and parameters
cat("Best SARIMA Model:\n")
print(best_model)
cat("Best Parameters:\n")
print(best_params)
cat("Best AIC:", best_aic, "\n")

library(astsa)
sarima(train_ts, 1, 1, 2, 1, 0, 0, 365)

#---------------------------Decompose for ETS(Frequency Too high)---------------------------------------------------------------------
# Decomposition with STL
monthlytrain_decompose <- stl(train_ts, s.window = "periodic")
plot(monthlytrain_decompose)

# Customized Decomposition Plots
components <- monthlytrain_decompose$time.series

# Plot trend, seasonality, and residuals separately
par(mfrow = c(3, 1)) # Multiple plots layout

# Extract the residuals from the decomposition
residuals <- monthlytrain_decompose$time.series[, "remainder"]
#---------------------------ETS(Frequency Too high)---------------------------------------------------------------------
# Fit a manually specified ETS model
manual_ets_model <- ets(train_ts, model = "ANA")
summary(manual_ets_model)