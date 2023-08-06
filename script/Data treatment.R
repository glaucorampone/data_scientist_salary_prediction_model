data <- read.csv("ds_salaries.csv")

# Eliminate salary features
data$salary_currency <- data$salary <- NULL

# Normalize remote_ratio
data$remote_ratio <- data$remote_ratio/100

# Store the names of columns to exclude from one-hot encoding
exclude_columns <- c("remote_ratio", "salary_in_usd")

# Extract the names of columns to be one-hot encoded
cols_to_encode <- setdiff(names(data), exclude_columns)

# Perform one-hot encoding for each column
for (col in cols_to_encode) {
  unique_values <- unique(data[[col]])
  for (val in unique_values)
    data[paste(col, "_", val, sep = "")] <- ifelse(data[[col]] == val, 1, 0)
}

# Remove the original columns as they are no longer needed
data <- data[, !names(data) %in% cols_to_encode]

# Split the dataset in Training, CV and Test set
# Set a seed for reproducibility
set.seed(123)

# Step 1: Shuffle the dataset randomly
shuffled_data <- data[sample(nrow(data)), ]

# Step 2: Define the proportions for each set (adjust these as needed)
train_ratio <- 0.7   # 70% of data for training
cv_ratio <- 0.15     # 15% of data for cross-validation
test_ratio <- 0.15   # 15% of data for test

# Calculate the number of samples for each set based on proportions
num_samples <- nrow(shuffled_data)
num_train <- round(train_ratio * num_samples)
num_cv <- round(cv_ratio * num_samples)

# Create the training, cross-validation, and test sets
train_set <- shuffled_data[1:num_train, ]
cv_set <- shuffled_data[(num_train + 1):(num_train + num_cv), ]
test_set <- shuffled_data[(num_train + num_cv + 1):num_samples, ]

# View the dimensions of each set
cat("Training set size:", nrow(train_set), "\n")
cat("Cross-validation set size:", nrow(cv_set), "\n")
cat("Test set size:", nrow(test_set), "\n")

# Save
write.csv(train_set, file.path(output_dir, "train_set.csv"), row.names = FALSE)
write.csv(cv_set, file.path(output_dir, "cv_set.csv"), row.names = FALSE)
write.csv(test_set, file.path(output_dir, "test_set.csv"), row.names = FALSE)

