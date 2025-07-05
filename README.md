# MovieLens Recommendation System — HarvardX Data Science Capstone

This repository contains my final capstone project for the HarvardX Data Science Professional Certificate. The objective was to build an accurate movie recommendation system using the MovieLens 10M dataset and apply techniques learned throughout the 9-course program.

## Project Structure

```
├── movielens_script.R          # Full project code with all steps and comments
├── movielens_report.Rmd        # RMarkdown report with narrative, code, and results
├── movielens_report.pdf        # Knit PDF version of the report
└── README.md                   # This file
```

## Objective

The goal was to predict how a user would rate a movie they haven't seen yet. This project involved:

- Data wrangling and preprocessing
- Exploratory data analysis
- Building baseline and advanced models
- Evaluating models using RMSE

## Methods Used

- Global Average Model
- Movie Effect Model
- Movie + User Effect Model
- Regularization to avoid overfitting
- Matrix Factorization using `recosystem` (an SVD-based recommendation library in R)

## Final Results

The final model used matrix factorization with the `recosystem` package and achieved:

- Final RMSE: below 0.8649, exceeding the benchmark required for full marks.

## Why recosystem?

While the course primarily focused on bias-based models, I wanted to go further. After researching and consulting with ChatGPT, I learned how to use the `recosystem` package for collaborative filtering. I implemented it myself step-by-step and am now confident using it in future projects.

## Reflections

This project helped me consolidate everything I’ve learned in the HarvardX program. I also stepped beyond the course material, explored a new recommendation algorithm, and improved the accuracy of my predictions.

## Dataset

- MovieLens 10M dataset: https://grouplens.org/datasets/movielens/10m/

## Author

Mayesha Maliha Proma  
Capstone Project for HarvardX PH125.9x

## License

This project is submitted as part of a course and is meant for educational purposes only.
