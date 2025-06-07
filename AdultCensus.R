################################################################################
# Adult Census Income Prediction Project
# HarvardX: PH125.9x Data Science: Capstone - Choose Your Own Project
# 
# Author: Mingyang Sun
# Date: 06.07.2025
# 
# Goal: Predict whether income exceeds $50K/year using census data
# Dataset: UCI Adult Census Income Dataset from Kaggle
# Models: Logistic Regression, Random Forest, and Gradient Boosting
################################################################################

# Load required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)
library(gbm)

################################################################################
# DATA LOADING AND PREPARATION
################################################################################

# Download and load the Adult Census Income dataset from Kaggle
cat("Downloading Adult Census Income dataset from Kaggle...\n")

temp_file <- "archive.zip"
if(!file.exists(temp_file))
  download.file("https://www.kaggle.com/api/v1/datasets/download/uciml/adult-census-income", temp_file)

# Extract and read the data files
unzip(temp_file, exdir = "data")

# Read the dataset (Kaggle version has headers)
combined_data <- read.csv("data/adult.csv", stringsAsFactors = FALSE, na.strings = c("?", " ?"))

# Clean up extracted files and zip
unlink("data", recursive = TRUE)
file.remove(temp_file)

cat("Dataset loaded successfully!\n")
cat("Combined dataset dimensions:", dim(combined_data), "\n")

################################################################################
# EXPLORATORY DATA ANALYSIS
################################################################################

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# Display basic information about the dataset
cat("Dataset Overview:\n")
cat("Number of observations:", nrow(combined_data), "\n")
cat("Number of variables:", ncol(combined_data), "\n")

# Check data structure
cat("\nData Structure:\n")
str(combined_data)

# Summary statistics
cat("\nSummary Statistics:\n")
summary(combined_data)

# Check for missing values
cat("\nMissing Values by Column:\n")
missing_counts <- sapply(combined_data, function(x) sum(is.na(x)))
missing_vars <- missing_counts[missing_counts > 0]
if(length(missing_vars) > 0) {
  for(i in 1:length(missing_vars)) {
    cat(names(missing_vars)[i], ":", missing_vars[i], 
        "(", round(missing_vars[i] / nrow(combined_data) * 100, 2), "%)\n")
  }
} else {
  cat("No missing values found.\n")
}

# Target variable distribution
cat("\nIncome Distribution:\n")
income_table <- table(combined_data$income)
print(income_table)
cat("Percentage earning >50K:", round(prop.table(income_table)[2] * 100, 2), "%\n")

# Age distribution by income
cat("\nAge Statistics by Income Level:\n")
age_by_income <- combined_data %>%
  group_by(income) %>%
  summarise(
    mean_age = mean(age, na.rm = TRUE),
    median_age = median(age, na.rm = TRUE),
    min_age = min(age, na.rm = TRUE),
    max_age = max(age, na.rm = TRUE),
    .groups = 'drop'
  )
print(age_by_income)

# Education level analysis
cat("\nEducation Level Distribution:\n")
education_income <- combined_data %>%
  group_by(education, income) %>%
  summarise(count = n(), .groups = 'drop') %>%
  spread(income, count, fill = 0)

if(ncol(education_income) >= 3) {
  education_income <- education_income %>%
    mutate(total = rowSums(select(., -education)),
           pct_high_income = round(get(names(.)[ncol(.)]) / total * 100, 1)) %>%
    arrange(desc(pct_high_income))
}
print(head(education_income, 10))

# Work hours analysis (adjust column name for Kaggle dataset)
hours_col <- ifelse("hours.per.week" %in% names(combined_data), "hours.per.week", "hours_per_week")
if(hours_col %in% names(combined_data)) {
  cat("\nWork Hours by Income Level:\n")
  hours_by_income <- combined_data %>%
    group_by(income) %>%
    summarise(
      mean_hours = mean(get(hours_col), na.rm = TRUE),
      median_hours = median(get(hours_col), na.rm = TRUE),
      .groups = 'drop'
    )
  print(hours_by_income)
}

################################################################################
# DATA CLEANING AND PREPROCESSING
################################################################################

cat("\n=== DATA PREPROCESSING ===\n")

# Create a copy for preprocessing
adult_clean <- combined_data

# Handle missing values
cat("Handling missing values...\n")

# Adjust column names for Kaggle dataset
categorical_vars <- c("workclass", "occupation")
if("native.country" %in% names(adult_clean)) {
  categorical_vars <- c(categorical_vars, "native.country")
} else if("native_country" %in% names(adult_clean)) {
  categorical_vars <- c(categorical_vars, "native_country")
}

for(var in categorical_vars) {
  if(var %in% names(adult_clean) && sum(is.na(adult_clean[[var]])) > 0) {
    mode_val <- names(sort(table(adult_clean[[var]]), decreasing = TRUE))[1]
    adult_clean[[var]][is.na(adult_clean[[var]])] <- mode_val
    cat("Replaced", sum(is.na(combined_data[[var]])), "missing values in", var, "with mode:", mode_val, "\n")
  }
}

# Convert income to binary factor (handle different formats)
adult_clean$income <- factor(ifelse(adult_clean$income == ">50K", 1, 0), 
                             levels = c(0, 1), labels = c("<=50K", ">50K"))

# Convert categorical variables to factors (adjust for Kaggle column names)
categorical_cols <- c("workclass", "education", "occupation", "relationship", "race", "sex")

# Add marital status column
if("marital.status" %in% names(adult_clean)) {
  categorical_cols <- c(categorical_cols, "marital.status")
} else if("marital_status" %in% names(adult_clean)) {
  categorical_cols <- c(categorical_cols, "marital_status")
}

# Add native country column
if("native.country" %in% names(adult_clean)) {
  categorical_cols <- c(categorical_cols, "native.country")
} else if("native_country" %in% names(adult_clean)) {
  categorical_cols <- c(categorical_cols, "native_country")
}

for(col in categorical_cols) {
  if(col %in% names(adult_clean)) {
    adult_clean[[col]] <- as.factor(adult_clean[[col]])
  }
}

# Feature engineering
cat("Creating new features...\n")

# Create age groups
adult_clean$age_group <- cut(adult_clean$age, 
                             breaks = c(0, 25, 35, 45, 55, 65, 100),
                             labels = c("18-25", "26-35", "36-45", "46-55", "56-65", "65+"))

# Create capital features (adjust for Kaggle column names)
capital_gain_col <- ifelse("capital.gain" %in% names(adult_clean), "capital.gain", "capital_gain")
capital_loss_col <- ifelse("capital.loss" %in% names(adult_clean), "capital.loss", "capital_loss")

adult_clean$has_capital_gain <- factor(ifelse(adult_clean[[capital_gain_col]] > 0, "Yes", "No"))
adult_clean$has_capital_loss <- factor(ifelse(adult_clean[[capital_loss_col]] > 0, "Yes", "No"))
adult_clean$net_capital <- adult_clean[[capital_gain_col]] - adult_clean[[capital_loss_col]]

# Work hours categories
hours_col <- ifelse("hours.per.week" %in% names(adult_clean), "hours.per.week", "hours_per_week")
adult_clean$work_hours_category <- cut(adult_clean[[hours_col]],
                                       breaks = c(0, 20, 40, 60, 100),
                                       labels = c("Part-time", "Full-time", "Overtime", "Excessive"))

# Education grouping
adult_clean$education_level <- case_when(
  adult_clean$education %in% c("Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", 
                               "10th", "11th", "12th") ~ "No_HS_Diploma",
  adult_clean$education %in% c("HS-grad", "Some-college") ~ "HS_Some_College",
  adult_clean$education %in% c("Assoc-voc", "Assoc-acdm") ~ "Associate",
  adult_clean$education == "Bachelors" ~ "Bachelors",
  adult_clean$education %in% c("Masters", "Prof-school", "Doctorate") ~ "Advanced"
)
adult_clean$education_level <- factor(adult_clean$education_level, 
                                      levels = c("No_HS_Diploma", "HS_Some_College", 
                                                 "Associate", "Bachelors", "Advanced"))

# Create aliases for variables with different names in Kaggle dataset
if("education.num" %in% names(adult_clean)) {
  adult_clean$education_num <- adult_clean$education.num
}
if("hours.per.week" %in% names(adult_clean)) {
  adult_clean$hours_per_week <- adult_clean$hours.per.week
}
if("marital.status" %in% names(adult_clean)) {
  adult_clean$marital_status <- adult_clean$marital.status
}
if("native.country" %in% names(adult_clean)) {
  adult_clean$native_country <- adult_clean$native.country
}
if("capital.gain" %in% names(adult_clean)) {
  adult_clean$capital_gain <- adult_clean$capital.gain
}
if("capital.loss" %in% names(adult_clean)) {
  adult_clean$capital_loss <- adult_clean$capital.loss
}

cat("Feature engineering completed.\n")
cat("Final dataset dimensions:", dim(adult_clean), "\n")

################################################################################
# DATA SPLITTING
################################################################################

cat("\n=== DATA SPLITTING ===\n")

# Set seed for reproducibility
set.seed(123)

# Create train/test split (80/20)
train_index <- createDataPartition(adult_clean$income, p = 0.8, list = FALSE)
train_data <- adult_clean[train_index, ]
test_data <- adult_clean[-train_index, ]

# Further split training data into train/validation (80/20 of training data)
val_index <- createDataPartition(train_data$income, p = 0.2, list = FALSE)
validation_data <- train_data[val_index, ]
train_data_final <- train_data[-val_index, ]

cat("Data split completed:\n")
cat("Training set:", nrow(train_data_final), "observations\n")
cat("Validation set:", nrow(validation_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")

# Check target distribution in splits
cat("\nTarget distribution in training set:\n")
print(prop.table(table(train_data_final$income)))

################################################################################
# MODEL DEVELOPMENT
################################################################################

cat("\n=== MODEL DEVELOPMENT ===\n")

# Define evaluation function
evaluate_model <- function(predictions, actual) {
  # Convert to factors with same levels if needed
  predictions <- factor(predictions, levels = levels(actual))
  
  # Confusion matrix
  cm <- confusionMatrix(predictions, actual)
  
  # Calculate metrics
  accuracy <- cm$overall['Accuracy']
  sensitivity <- cm$byClass['Sensitivity']  # True Positive Rate
  specificity <- cm$byClass['Specificity']  # True Negative Rate
  precision <- cm$byClass['Pos Pred Value']
  f1_score <- cm$byClass['F1']
  
  return(list(
    confusion_matrix = cm,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1_score
  ))
}

# Store results
model_results <- data.frame()

################################################################################
# Model 1: Logistic Regression
################################################################################

cat("\nTraining Logistic Regression Model...\n")

# Prepare data for logistic regression (select key variables to avoid overfitting)
key_vars <- c("age", "education_num", "hours_per_week", "sex", "marital_status", 
              "education_level", "work_hours_category", "has_capital_gain", "net_capital")

# Create formula
formula_lr <- as.formula(paste("income ~", paste(key_vars, collapse = " + ")))

# Train logistic regression
lr_model <- glm(formula_lr, data = train_data_final, family = "binomial")

# Make predictions on validation set
lr_pred_prob <- predict(lr_model, validation_data, type = "response")
lr_pred <- factor(ifelse(lr_pred_prob > 0.5, ">50K", "<=50K"), 
                  levels = c("<=50K", ">50K"))

# Evaluate
lr_eval <- evaluate_model(lr_pred, validation_data$income)
cat("Logistic Regression - Validation Accuracy:", round(lr_eval$accuracy, 4), "\n")

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Logistic Regression",
  Accuracy = lr_eval$accuracy,
  Sensitivity = lr_eval$sensitivity,
  Specificity = lr_eval$specificity,
  F1_Score = lr_eval$f1_score
))

################################################################################
# Model 2: Random Forest
################################################################################

cat("\nTraining Random Forest Model...\n")

# Prepare data for Random Forest (use more variables)
rf_vars <- c("age", "workclass", "education_num", "marital_status", "occupation",
             "relationship", "race", "sex", "hours_per_week", "education_level",
             "work_hours_category", "has_capital_gain", "has_capital_loss", "net_capital")

# Create training data subset
train_rf <- train_data_final[, c(rf_vars, "income")]
validation_rf <- validation_data[, c(rf_vars, "income")]

# Train Random Forest
set.seed(123)
rf_model <- randomForest(income ~ ., data = train_rf, 
                         ntree = 500, mtry = 4, importance = TRUE)

# Make predictions
rf_pred <- predict(rf_model, validation_rf)

# Evaluate
rf_eval <- evaluate_model(rf_pred, validation_data$income)
cat("Random Forest - Validation Accuracy:", round(rf_eval$accuracy, 4), "\n")

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Random Forest",
  Accuracy = rf_eval$accuracy,
  Sensitivity = rf_eval$sensitivity,
  Specificity = rf_eval$specificity,
  F1_Score = rf_eval$f1_score
))

# Variable importance
cat("Top 10 Most Important Variables (Random Forest):\n")
importance_rf <- importance(rf_model)
importance_df <- data.frame(
  Variable = rownames(importance_rf),
  MeanDecreaseGini = importance_rf[, "MeanDecreaseGini"]
) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(10)
print(importance_df)

################################################################################
# Model 3: Gradient Boosting Machine (GBM)
################################################################################

cat("\nTraining Gradient Boosting Model...\n")

# Prepare data for GBM
train_gbm <- train_data_final[, c(rf_vars, "income")]
validation_gbm <- validation_data[, c(rf_vars, "income")]

# Convert income to numeric for GBM (0/1)
train_gbm$income_numeric <- as.numeric(train_gbm$income) - 1
validation_gbm$income_numeric <- as.numeric(validation_gbm$income) - 1

# Train GBM
set.seed(123)
gbm_model <- gbm(income_numeric ~ . - income, data = train_gbm,
                 distribution = "bernoulli", n.trees = 1000, 
                 interaction.depth = 4, shrinkage = 0.01, cv.folds = 5)

# Find optimal number of trees
optimal_trees <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
cat("Optimal number of trees:", optimal_trees, "\n")

# Make predictions
gbm_pred_prob <- predict(gbm_model, validation_gbm, n.trees = optimal_trees, type = "response")
gbm_pred <- factor(ifelse(gbm_pred_prob > 0.5, ">50K", "<=50K"), 
                   levels = c("<=50K", ">50K"))

# Evaluate
gbm_eval <- evaluate_model(gbm_pred, validation_data$income)
cat("Gradient Boosting - Validation Accuracy:", round(gbm_eval$accuracy, 4), "\n")

# Store results
model_results <- rbind(model_results, data.frame(
  Model = "Gradient Boosting",
  Accuracy = gbm_eval$accuracy,
  Sensitivity = gbm_eval$sensitivity,
  Specificity = gbm_eval$specificity,
  F1_Score = gbm_eval$f1_score
))

# Variable importance for GBM
cat("Top 10 Most Important Variables (GBM):\n")
importance_gbm <- summary(gbm_model, n.trees = optimal_trees, plotit = FALSE)
print(head(importance_gbm, 10))

################################################################################
# MODEL COMPARISON AND SELECTION
################################################################################

cat("\n=== MODEL COMPARISON ===\n")
# Display results with proper formatting
cat("Model Performance Comparison:\n")
cat("Model\t\t\tAccuracy\tSensitivity\tSpecificity\tF1_Score\n")
for(i in 1:nrow(model_results)) {
  cat(sprintf("%-20s\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", 
              model_results$Model[i], 
              model_results$Accuracy[i],
              model_results$Sensitivity[i], 
              model_results$Specificity[i],
              model_results$F1_Score[i]))
}

# Select best model based on accuracy
best_model_idx <- which.max(model_results$Accuracy)
best_model_name <- model_results$Model[best_model_idx]
cat("\nBest performing model:", best_model_name, "\n")

################################################################################
# FINAL MODEL EVALUATION ON TEST SET
################################################################################

cat("\n=== FINAL EVALUATION ON TEST SET ===\n")

# Use the best performing model for final evaluation
if(best_model_name == "Logistic Regression") {
  final_pred_prob <- predict(lr_model, test_data, type = "response")
  final_pred <- factor(ifelse(final_pred_prob > 0.5, ">50K", "<=50K"), 
                       levels = c("<=50K", ">50K"))
} else if(best_model_name == "Random Forest") {
  test_rf <- test_data[, c(rf_vars, "income")]
  final_pred <- predict(rf_model, test_rf)
} else if(best_model_name == "Gradient Boosting") {
  test_gbm <- test_data[, c(rf_vars, "income")]
  final_pred_prob <- predict(gbm_model, test_gbm, n.trees = optimal_trees, type = "response")
  final_pred <- factor(ifelse(final_pred_prob > 0.5, ">50K", "<=50K"), 
                       levels = c("<=50K", ">50K"))
} else {
  # Fallback to Random Forest if no match
  test_rf <- test_data[, c(rf_vars, "income")]
  final_pred <- predict(rf_model, test_rf)
}

# Final evaluation
final_eval <- evaluate_model(final_pred, test_data$income)

cat("FINAL RESULTS ON TEST SET:\n")
cat("Model:", best_model_name, "\n")
cat("Accuracy:", round(final_eval$accuracy, 4), "\n")
cat("Sensitivity (True Positive Rate):", round(final_eval$sensitivity, 4), "\n")
cat("Specificity (True Negative Rate):", round(final_eval$specificity, 4), "\n")
cat("Precision:", round(final_eval$precision, 4), "\n")
cat("F1 Score:", round(final_eval$f1_score, 4), "\n")

# Display confusion matrix
cat("\nConfusion Matrix:\n")
print(final_eval$confusion_matrix$table)

# Calculate additional metrics for final reporting
final_accuracy <- final_eval$accuracy
final_sensitivity <- final_eval$sensitivity
final_specificity <- final_eval$specificity
final_f1 <- final_eval$f1_score

cat("PROJECT COMPLETED SUCCESSFULLY\n")
cat("Final Model:", best_model_name, "\n")
cat("Test Set Performance:\n")
cat("- Accuracy:", round(final_accuracy, 4), "\n")
cat("- Sensitivity:", round(final_sensitivity, 4), "\n") 
cat("- Specificity:", round(final_specificity, 4), "\n")
cat("- F1 Score:", round(final_f1, 4), "\n")
