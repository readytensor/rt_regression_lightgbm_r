# Required Libraries
library(tidyverse)
library(jsonlite)
library(fastDummies)
library(lightgbm)

# Paths
ROOT_DIR <- dirname(getwd())
MODEL_INPUTS_OUTPUTS <- file.path(ROOT_DIR, 'model_inputs_outputs')
INPUT_DIR <- file.path(MODEL_INPUTS_OUTPUTS, "inputs")
OUTPUT_DIR <- file.path(MODEL_INPUTS_OUTPUTS, "outputs")
INPUT_SCHEMA_DIR <- file.path(INPUT_DIR, "schema")
DATA_DIR <- file.path(INPUT_DIR, "data")
TRAIN_DIR <- file.path(DATA_DIR, "training")
TEST_DIR <- file.path(DATA_DIR, "testing")
MODEL_PATH <- file.path(MODEL_INPUTS_OUTPUTS, "model")
MODEL_ARTIFACTS_PATH <- file.path(MODEL_PATH, "artifacts")
OHE_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'ohe.rds')
PREDICTOR_DIR_PATH <- file.path(MODEL_ARTIFACTS_PATH, "predictor")
PREDICTOR_FILE_PATH <- file.path(PREDICTOR_DIR_PATH, "predictor.rds")
IMPUTATION_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'imputation.rds')
PREDICTIONS_DIR <- file.path(OUTPUT_DIR, 'predictions')
PREDICTIONS_FILE <- file.path(PREDICTIONS_DIR, 'predictions.csv')
TOP_10_CATEGORIES_MAP <- file.path(MODEL_ARTIFACTS_PATH, "top_10_map.rds")
COLNAME_MAPPING <- file.path(MODEL_ARTIFACTS_PATH, "colname_mapping.csv")
SCALING_FILE <- file.path(MODEL_ARTIFACTS_PATH, "scaler.rds")


if (!dir.exists(PREDICTIONS_DIR)) {
  dir.create(PREDICTIONS_DIR, recursive = TRUE)
}

# Reading the schema
file_name <- list.files(INPUT_SCHEMA_DIR, pattern = "*.json")[1]
schema <- fromJSON(file.path(INPUT_SCHEMA_DIR, file_name))
features <- schema$features

numeric_features <- features$name[features$dataType != 'CATEGORICAL']
categorical_features <- features$name[features$dataType == 'CATEGORICAL']
id_feature <- schema$id$name
target_feature <- schema$target$name
target_classes <- schema$target$classes
model_category <- schema$modelCategory
nullable_features <- features$name[features$nullable == TRUE]


# Reading test data.
file_name <- list.files(TEST_DIR, pattern = "*.csv", full.names = TRUE)[1]
# Read the first line to get column names
header_line <- readLines(file_name, n = 1)
col_names <- unlist(strsplit(header_line, split = ",")) # assuming ',' is the delimiter
# Read the CSV with the exact column names
df <- read.csv(file_name, skip = 0, col.names = col_names, check.names=FALSE)


# Data preprocessing
# Note that when we work with testing data, we have to impute using the same values learned during training. This is to avoid data leakage.
imputation_values <- readRDS(IMPUTATION_FILE)

for (column in nullable_features) {
    # Create missing indicator
    missing_indicator_col_name <- paste(column, "is_missing", sep="_")
    df[[missing_indicator_col_name]] <- ifelse(is.na(df[[column]]), 1, 0)
    
    # Impute missing values
    if (!is.null(imputation_values[[column]])) {
        df[, column][is.na(df[, column])] <- imputation_values[[column]]
    }
}

# Saving the id column in a different variable and then dropping it.
ids <- df[[id_feature]]
df[[id_feature]] <- NULL

# Encoding
# We encode the data using the same encoder that we saved during training.
if (length(categorical_features) > 0 && file.exists(OHE_ENCODER_FILE)) {
  top_10_map <- readRDS(TOP_10_CATEGORIES_MAP)
  encoder <- readRDS(OHE_ENCODER_FILE)
  for(col in categorical_features) {
    # Use the saved top 10 categories to replace values outside these categories with 'Other'
    df[[col]][!(df[[col]] %in% top_10_map[[col]])] <- "Other"
  }

  test_df_encoded <- dummy_cols(df, select_columns = categorical_features, remove_selected_columns = TRUE)
  encoded_columns <- readRDS(OHE_ENCODER_FILE)
  # Add missing columns with 0s
    for (col in encoded_columns) {
        if (!col %in% colnames(test_df_encoded)) {
            test_df_encoded[[col]] <- 0
        }
    }

# Remove extra columns
    extra_cols <- setdiff(colnames(test_df_encoded), c(colnames(df), encoded_columns))
    df <- test_df_encoded[, !names(test_df_encoded) %in% extra_cols]
}

# Standard Scaling
scaling_values <- readRDS(SCALING_FILE) # Assuming you've saved scaling values during training
for (feature in numeric_features) {
    df[[feature]] <- (df[[feature]] - scaling_values[[feature]]$mean) / scaling_values[[feature]]$std
}

# Outlier Capping for Standard Scaled Data
lower_bound <- -4
upper_bound <- 4

for (feature in numeric_features) {
    df[[feature]] <- ifelse(df[[feature]] < lower_bound, lower_bound, df[[feature]])
    df[[feature]] <- ifelse(df[[feature]] > upper_bound, upper_bound, df[[feature]])
}

# Load the column name mapping
colname_mapping <- read.csv(COLNAME_MAPPING)
# Update the column names based on the mapping
df <- df[, colname_mapping$original]
colnames(df) <- colname_mapping$sanitized[match(colnames(df), colname_mapping$original)]


# Making predictions
# Load the trained LightGBM model
model <- lgb.load(PREDICTOR_FILE_PATH)

# Predict using the LightGBM model
# Note: For LightGBM, the prediction function requires a matrix. 
predictions <- predict(model, data = as.matrix(df))

# Round the predictions to 5 decimals
predictions <- round(predictions, 5)

# Creating predictions DataFrame
predictions_df <- data.frame(prediction = predictions)
predictions_df <- tibble::tibble(ids = ids) %>% dplyr::bind_cols(predictions_df)
colnames(predictions_df)[1] <- id_feature

# avoid using scientific notation when writing the CSV file
options(scipen = 999)

# write the predictions to a CSV file
write.csv(predictions_df, PREDICTIONS_FILE, row.names = FALSE)

