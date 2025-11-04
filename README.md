# Student Grade Predictor

## Overview

The Student Grade Predictor is an interactive Python tool designed to help students plan and forecast their academic performance. It combines an optimizer to calculate minimum required scores and a multi-output linear regression model to predict expected scores for incomplete assessments.

## Features

* **Minimum Score Optimizer:** Determines the scores needed on pending assessments to reach a user-defined target final grade.
* **Regression Predictor:** Estimates likely scores for incomplete assessments based on current performance and historical data.
* **Continuous Learning:** Updates training data with completed assessments to improve prediction accuracy over time.
* **Interactive Input:** Handles any number of assessments, including their weights and current scores.
* **Multi-Assessment Support:** Each assessment has a separate regressor for more accurate predictions.

## Tech Stack

* Python 3
* scikit-learn (LinearRegression)
* NumPy
* JSON (for persistent training data storage)

## Usage

1. Run the script using Python: `python grade_predictor.py`
2. Enter your target final grade.
3. Input the number of assessments along with their names, weights, and scores.
4. The program outputs:

   * Minimum scores needed to meet the target grade.
   * Predicted scores for incomplete assessments.
   * Predicted final grade based on regression estimates.

## Example

```
Enter your target final grade (e.g., 85): 90
Enter number of assessments: 2
Enter name of assessment 1: Test 1
Enter weight of Test 1 (%): 40
Enter score in Test 1 (or leave blank if not done): 90
Enter name of assessment 2: Test 2
Enter weight of Test 2 (%): 60
Enter score in Test 2 (or leave blank if not done): 

Result:
Minimum required score for Test 2: 90%
Predicted score for Test 2: 84%
Predicted final grade: 86.5%
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, improvements, or new features.

## License

This project is open-source and available under the Apache 2.0 License.
