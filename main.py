# Student Grade Predictor with Continuous-Learning Multi-Assessment Linear Regression Estimation

from sklearn.linear_model import LinearRegression
import numpy as np
import json
import os

DATA_FILE = "training_data.json"


def load_training_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []


def save_training_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)


def calculate_required_scores(target, assessments):
    known_score = 0
    remaining_weight = 0
    remaining = []

    for a in assessments:
        w = a['weight']
        s = a['score']
        if s is not None:
            known_score += (s * w) / 100
        else:
            remaining_weight += w
            remaining.append(a)

    needed = target - known_score

    if remaining_weight == 0:
        return {"message": f"All assessments completed. Target {'met' if needed <= 0 else 'not met'}.", "required": {}}

    req_score = (needed * 100) / remaining_weight

    if req_score > 100:
        return {"message": "Target is unreachable with remaining assessments.", "required": {}}

    required = {}
    for a in remaining:
        required[a['name']] = round(req_score, 2)

    return {"message": "Calculated required scores successfully.", "required": required}


def train_multi_regressors(data, n_assessments):
    regressors = []

    for i in range(n_assessments):
        X = []
        y = []
        for features, scores in data:
            if len(features) < n_assessments or len(scores) < n_assessments:
                continue
            X.append([s for j, s in enumerate(features) if j != i])
            y.append(scores[i])
        if X and y:
            model = LinearRegression()
            model.fit(X, y)
            regressors.append(model)
        else:
            regressors.append(None)

    return regressors


def run_parallel_prediction(assessments, target, data):
    print("\n--- Parallel Predictor & Multi-Regressor Model ---")

    n_assessments = len(assessments)

    # Combine saved and default training data
    default_data = [
        ([80, 70], [80, 70]),
        ([90, 85], [90, 85]),
        ([60, 65], [60, 65]),
        ([70, 75], [70, 75]),
        ([85, 80], [85, 80]),
        ([50, 55], [50, 55]),
        ([95, 90], [95, 90]),
    ] if n_assessments == 2 else [
        ([80, 70, 75], [80, 70, 75]),
        ([90, 85, 88], [90, 85, 88]),
        ([60, 65, 70], [60, 65, 70]),
        ([70, 75, 80], [70, 75, 80]),
        ([85, 80, 90], [85, 80, 90]),
        ([50, 55, 60], [50, 55, 60]),
        ([95, 90, 92], [95, 90, 92]),
    ]

    combined_data = default_data + data
    regressors = train_multi_regressors(combined_data, n_assessments)

    scores = [a['score'] for a in assessments]

    predicted_scores = []
    for i, s in enumerate(scores):
        if s is not None:
            predicted_scores.append(s)
        else:
            if regressors[i] is None:
                predicted_scores.append(70.0)  # fallback if not enough data
                continue

            features = [v for j, v in enumerate(scores) if j != i and v is not None]

            # Fill missing values in feature vector with average known values
            if not features:
                predicted_scores.append(70.0)
                continue

            avg_value = np.mean(features)
            X_input = [v if v is not None else avg_value for v in features]

            try:
                predicted = regressors[i].predict([X_input])[0]
            except Exception:
                predicted = avg_value

            predicted_scores.append(round(predicted, 2))

    predicted_final = sum((predicted_scores[i] * a['weight']) / 100 for i, a in enumerate(assessments))

    print("\n--- Predictor Results ---")
    calc = calculate_required_scores(target, assessments)
    if calc['required']:
        for name, req in calc['required'].items():
            print(f"To meet target {target}%, you need at least {req}% in {name}.")
    else:
        print(calc['message'])

    print("\n--- Regression Estimator Results ---")
    for i, a in enumerate(assessments):
        print(f"Predicted {a['name']} score: {predicted_scores[i]}%")
    print(f"Predicted Final Grade (Regression): {predicted_final:.2f}%")

    # Add user data for continuous learning
    if all(a['score'] is not None for a in assessments):
        features = [a['score'] for a in assessments]
        data.append((features, features))
        save_training_data(data)
        print("\nTraining data updated for continuous learning!")


def run_interactive():
    print("--- Student Grade Predictor (Continuous Learning Mode) ---")

    try:
        target = float(input("Enter your target final grade (e.g., 75): "))
        num_assessments = int(input("Enter number of assessments: "))

        assessments = []
        for i in range(num_assessments):
            name = input(f"Enter name of assessment {i + 1}: ")
            weight = float(input(f"Enter weight of {name} (%): "))
            score_input = input(f"Enter score in {name} (or leave blank if not done): ")
            score = float(score_input) if score_input else None
            assessments.append({"name": name, "weight": weight, "score": score})

        user_data = load_training_data()
        run_parallel_prediction(assessments, target, user_data)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_interactive()
