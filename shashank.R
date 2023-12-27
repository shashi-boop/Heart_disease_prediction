setwd("C:\\Users\\shash\\OneDrive\\Desktop\\Data Science and Analytics")
###################################### Question 2 ###########################
####### a) to calculate total number of parameters
# Load the neuralnet library
#install.packages("neuralnet")
library(neuralnet)
set.seed(414)  

# sample number of neurons for the input and hidden layers
layer_neurons_input_hidden <- sample(3:10, 2)
input_neurons <- layer_neurons_input_hidden[1]
hidden_neurons <- layer_neurons_input_hidden[2]

# sample number of neurons for the output layer
output_neurons <- sample(2:4, 1)

# Print out the sampled number of neurons for each layer
cat("Number of neurons in input layer:", input_neurons, "\n")
cat("Number of neurons in hidden layer:", hidden_neurons, "\n")
cat("Number of neurons in output layer:", output_neurons, "\n")




##################################### question 3 ############################
# read.dat file with headers
data <- read.table("heart.dat", sep=" ", header=FALSE)
# copying the data into another variable
data1 = data
# structure of the data
str(data)
# Assigning column names
colnames(data) <- c(
  "age", "sex", "chest_pain_type", "resting_blood_pressure",
  "serum_cholesterol", "fasting_blood_sugar", "resting_ecg",
  "max_heart_rate", "exercise_induced_angina", "oldpeak",
  "slope", "num_major_vessels", "thal", "heart_disease"
)

# Checking the structure
str(data)
##################### EDA(EXPLOROtary DATA ANALYSIS) #####################
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(corrplot)

# Check the structure of the data
str(data)

# Summary Statistics
summary(data)

# Data Visualization

# Histograms for numerical variables
num_vars <- c('age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate', 'oldpeak', 'slope', 'num_major_vessels')
ggplot(data, aes(x=age, y=serum_cholesterol, color=factor(heart_disease))) +
  geom_point() +
  geom_smooth(method='lm', se=FALSE, aes(group=factor(heart_disease), color=factor(heart_disease))) +
  ggtitle("Age vs Serum Cholesterol conditioned on Heart Disease")


ggplot(data, aes(x=age, y=max_heart_rate, color=factor(heart_disease))) +
  geom_point() +
  geom_smooth(method='lm', se=FALSE, aes(group=factor(heart_disease), color=factor(heart_disease))) +
  ggtitle("Age vs Max Heart Rate conditioned on Heart Disease")

# Boxplots for Age against different categorical variables
ggplot(data, aes(x=factor(sex), y=age, fill=factor(sex))) +
  geom_boxplot() +
  ggtitle("Boxplot of Age by Sex")

ggplot(data, aes(x=factor(chest_pain_type), y=age, fill=factor(chest_pain_type))) +
  geom_boxplot() +
  ggtitle("Boxplot of Age by Chest Pain Type")


# Correlation Analysis
cor_matrix <- cor(data[, num_vars], method='pearson')
corrplot(cor_matrix, method='circle')

# Class Distribution
table(data$heart_disease)

# Check for missing values
sum(is.na(data))

##################### b ##############################
# Normalize data
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)

scaled_data <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

# Split the data into training (80%) and testing (20%) sets
set.seed(414)
sample <- sample(1:nrow(scaled_data), size = 0.8 * nrow(scaled_data))
train_data <- scaled_data[sample, ]
test_data <- scaled_data[-sample, ]
# Install and load the neuralnet library
install.packages("neuralnet")
library(neuralnet)

# Define the formula for predicting 'heart_disease' based on other variables
formula <- as.formula("heart_disease ~ age + sex + chest_pain_type + resting_blood_pressure + serum_cholesterol + fasting_blood_sugar + resting_ecg + max_heart_rate + exercise_induced_angina + oldpeak + slope + num_major_vessels + thal")

# Train the neural network
nn <- neuralnet(formula, data = train_data, hidden = c(5, 5), linear.output = FALSE, act.fct = "logistic")

# Plot the neural network
plot(nn)

################################ c #####################
# Compute the neural network predictions
nn_predictions <- predict(nn, test_data[,1:13])

# Convert these probabilities to binary outcomes (assuming threshold = 0.5)
nn_pred_class <- ifelse(nn_predictions > 0.5, 1, 0)

# Create a confusion matrix
library(caret) # Load the 'caret' package for confusionMatrix function
conf_matrix <- confusionMatrix(as.factor(nn_pred_class), as.factor(test_data$heart_disease))

# Print the confusion matrix
print(conf_matrix)

############################# e #################################
# Standardizing the data by dividing each value by the max value of each column
std_data <- as.data.frame(lapply(data, function(x) x / max(x, na.rm = TRUE)))

# Splitting the data into training (80%) and test (20%) sets
set.seed(414)
sample <- sample(1:nrow(std_data), size = 0.8 * nrow(std_data))
train_data_std <- std_data[sample, ]
test_data_std <- std_data[-sample, ]

# Train the neural network on standardized data
nn_std <- neuralnet(formula, data = train_data_std, hidden = c(5, 5), linear.output = FALSE, act.fct = "logistic")

plot(nn_std)
# Make predictions on the test set
nn_predictions_std <- predict(nn_std, test_data_std[,1:13])
nn_pred_class_std <- ifelse(nn_predictions_std > 0.5, 1, 0)

# Create a confusion matrix for the standardized model
conf_matrix_std <- confusionMatrix(as.factor(nn_pred_class_std), as.factor(test_data_std$heart_disease))
print(conf_matrix_std)

