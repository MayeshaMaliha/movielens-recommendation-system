# MovieLens Capstone Project: Final R Script
# Author: Mayesha Maliha Proma
# Goal: Build and evaluate a movie recommendation system using matrix factorization (recosystem)

# -------------------------
# Load required libraries
# -------------------------
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(recosystem)

# -------------------------
# Load and clean the data
# -------------------------
ratings_file <- "/Users/mayeshamalihaproma/Downloads/ml-10M100K/ratings.dat"
movies_file  <- "/Users/mayeshamalihaproma/Downloads/ml-10M100K/movies.dat"

# Read the raw ratings data file, where each line has 4 fields separated by "::"
ratings <- read_lines(ratings_file) %>%
  str_split_fixed("::", 4) %>%
  as.data.frame(stringsAsFactors = FALSE)

# Rename the columns to meaningful names
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

# Convert each column to the appropriate data type:
ratings <- ratings %>%
  mutate(userId = as.integer(userId), # - userId and movieId should be integers
         movieId = as.integer(movieId), 
         rating = as.numeric(rating), # - rating should be numeric (decimal)
         timestamp = as.integer(timestamp)) # - timestamp should be an integer (Unix time)

# Read the raw movies data file, where each line has 3 fields separated by "::"
movies <- read_lines(movies_file) %>%
  str_split_fixed("::", 3) %>%
  as.data.frame(stringsAsFactors = FALSE)

# Assign proper column names: movieId, title, and genres
colnames(movies) <- c("movieId", "title", "genres")

# Convert movieId to integer so it can be joined with the ratings data later
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merge the ratings and movies data frames using movieId as the common key
movielens <- left_join(ratings, movies, by = "movieId")

# -------------------------
# Split into edx and final_holdout_test
# -------------------------

# Set seed for reproducibility
set.seed(1, sample.kind = "Rounding")

# Create a 90/10 split of movielens ratings: 90% for edx, 10% for validation
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Keep only rows in the holdout set where both userId and movieId exist in edx
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Put any excluded rows (due to missing users/movies) back into edx
removed <- anti_join(temp, final_holdout_test)
edx <- bind_rows(edx, removed)

# Remove temporary variables from the environment to clean up memory
rm(ratings, movies, test_index, temp, removed)

# -------------------------
# Define RMSE function
# -------------------------

# Define a function to compute Root Mean Squared Error (RMSE)
# RMSE is a common metric to evaluate prediction accuracy in recommendation systems
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# -------------------------
# Train Matrix Factorization model using recosystem
# -------------------------

# Prepare training data from edx: only userId, movieId, and rating columns
train_data <- edx %>%
  select(userId, movieId, rating)

# Prepare test data from final_holdout_test: userId and movieId only
test_data <- final_holdout_test %>%
  select(userId, movieId)

# Create temporary file paths to store data for recosystem
train_file <- tempfile()
test_file <- tempfile()
out_file <- tempfile()

# Write the training data to a space-separated file with no row/column names
write.table(train_data, file = train_file, sep = " ", row.names = FALSE, col.names = FALSE)

# Write the test data similarly
write.table(test_data, file = test_file, sep = " ", row.names = FALSE, col.names = FALSE)

# Initialize the recosystem model object
r <- Reco()

# Tune the model to find the best hyperparameters (dim, learning rate, regularization)
opts <- r$tune(train_file, opts = list(
  dim = c(10, 20, 30),         # Number of latent features
  lrate = c(0.1, 0.2),         # Learning rates to try
  costp_l2 = c(0.01, 0.1),     # Regularization for user features
  costq_l2 = c(0.01, 0.1),     # Regularization for item features
  nthread = 4,                 # Use 4 CPU threads
  niter = 10                   # Run 10 iterations during tuning
))

# Train the model on the entire edx dataset using the best parameters found above
r$train(train_file, opts = c(opts$min, nthread = 4, niter = 20))

# Predict ratings on the final_holdout_test set
r$predict(test_file, out_file)

# Read the predicted ratings from the output file into R
predicted_ratings <- scan(out_file)

# -------------------------
# Compute final RMSE
# -------------------------

# Calculate and store the RMSE between actual and predicted ratings
final_rmse <- RMSE(final_holdout_test$rating, predicted_ratings)

# Print the final RMSE clearly
cat("Final RMSE from recosystem model:", final_rmse, "\n")