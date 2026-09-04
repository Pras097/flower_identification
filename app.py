from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load Iris dataset
iris = load_iris()

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

# Information about each Iris species
flower_info = {
    0: {
        "name": "Iris Setosa",
        "icon": "🌸",
        "description": "Iris Setosa usually has small petals and is clearly different from the other Iris species."
    },
    1: {
        "name": "Iris Versicolor",
        "icon": "🌼",
        "description": "Iris Versicolor has medium-sized petals and sepals with measurements between Setosa and Virginica."
    },
    2: {
        "name": "Iris Virginica",
        "icon": "🌺",
        "description": "Iris Virginica generally has larger petals and sepals compared with the other Iris species."
    }
}


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    probabilities = None
    error = None

    values = {
        "sepal_length": "",
        "sepal_width": "",
        "petal_length": "",
        "petal_width": ""
    }

    if request.method == "POST":

        try:
            # Get values from the form
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            values = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }

            # Prepare input
            input_data = [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]]

            # Prediction
            prediction = int(model.predict(input_data)[0])

            # Prediction probabilities
            probability_values = model.predict_proba(input_data)[0]

            probabilities = {
                "Iris-Setosa": round(probability_values[0] * 100, 2),
                "Iris-Versicolor": round(probability_values[1] * 100, 2),
                "Iris-Virginica": round(probability_values[2] * 100, 2)
            }

            # Confidence
            confidence = round(max(probability_values) * 100, 2)

        except (ValueError, KeyError):
            error = "Please enter valid numerical values for all four measurements."

        except Exception:
            error = "Something went wrong while making the prediction."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        result_info=flower_info.get(prediction),
        values=values,
        error=error,
        model_loaded=True
    )


if __name__ == "__main__":
    app.run(debug=True)