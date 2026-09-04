from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import os
import math
import pickle

try:
    import joblib
except ImportError:
    joblib = None


app = Flask(__name__)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = "iris_model.pkl"


def load_model():

    # First try to load your existing iris_model.pkl
    if os.path.exists(MODEL_PATH):

        # Try joblib first
        if joblib is not None:

            try:
                return joblib.load(MODEL_PATH)
            except Exception:
                pass

        # Try normal pickle
        try:

            with open(MODEL_PATH, "rb") as file:
                return pickle.load(file)

        except Exception:
            pass


    # --------------------------------------------------------
    # FALLBACK MODEL
    # --------------------------------------------------------

    iris = load_iris()

    model = LogisticRegression(
        max_iter=200
    )

    model.fit(
        iris.data,
        iris.target
    )

    return model


try:

    model = load_model()

    MODEL_READY = True

except Exception as e:

    model = None

    MODEL_READY = False

    print("Model loading error:", e)


# ============================================================
# IRIS SPECIES INFORMATION
# ============================================================

flower_info = {

    "Iris-setosa": {

        "display_name": "Iris Setosa",

        "scientific_name": "Iris setosa",

        "symbol": "🌸",

        "description":
            "Iris Setosa is a perennial flowering plant "
            "belonging to the Iridaceae family. It is generally "
            "recognized by its smaller petals and broader sepals. "
            "In the Iris dataset, it usually has relatively small "
            "petal measurements."

    },


    "Iris-versicolor": {

        "display_name": "Iris Versicolor",

        "scientific_name": "Iris versicolor",

        "symbol": "🌼",

        "description":
            "Iris Versicolor is a perennial flowering plant "
            "belonging to the Iridaceae family. It is commonly "
            "associated with blue to violet flowers and generally "
            "has intermediate measurements compared with "
            "Iris Setosa and Iris Virginica."

    },


    "Iris-virginica": {

        "display_name": "Iris Virginica",

        "scientific_name": "Iris virginica",

        "symbol": "🌺",

        "description":
            "Iris Virginica is a perennial flowering plant "
            "belonging to the Iridaceae family. It generally "
            "has larger flowers and comparatively longer and "
            "wider petals. In the Iris dataset, it commonly "
            "has larger measurements than the other two species."

    }

}


# ============================================================
# CONVERT MODEL OUTPUT TO STANDARD SPECIES NAME
# ============================================================

def get_species_name(prediction):

    """
    Converts different possible model outputs into the
    standard names used by index.html.
    """

    # If prediction is numeric
    try:

        prediction_number = int(prediction)

        numeric_species = {

            0: "Iris-setosa",

            1: "Iris-versicolor",

            2: "Iris-virginica"

        }

        if prediction_number in numeric_species:

            return numeric_species[prediction_number]

    except (ValueError, TypeError):

        pass


    # If model already returns a string
    if isinstance(prediction, str):

        clean_prediction = prediction.strip().lower()

        string_species = {

            "iris-setosa": "Iris-setosa",

            "iris setosa": "Iris-setosa",

            "setosa": "Iris-setosa",

            "iris-versicolor": "Iris-versicolor",

            "iris versicolor": "Iris-versicolor",

            "versicolor": "Iris-versicolor",

            "iris-virginica": "Iris-virginica",

            "iris virginica": "Iris-virginica",

            "virginica": "Iris-virginica"

        }

        return string_species.get(
            clean_prediction,
            prediction
        )


    return str(prediction)


# ============================================================
# GET STANDARD LABEL FROM MODEL CLASS
# ============================================================

def standardize_class_name(class_value):

    return get_species_name(class_value)


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    error = None

    result = None

    values = {

        "sepal_length": "",

        "sepal_width": "",

        "petal_length": "",

        "petal_width": ""

    }


    # ========================================================
    # HANDLE FORM SUBMISSION
    # ========================================================

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # GET FORM VALUES
            # ------------------------------------------------

            sepal_length_text = request.form.get(
                "sepal_length",
                ""
            ).strip()


            sepal_width_text = request.form.get(
                "sepal_width",
                ""
            ).strip()


            petal_length_text = request.form.get(
                "petal_length",
                ""
            ).strip()


            petal_width_text = request.form.get(
                "petal_width",
                ""
            ).strip()


            # ------------------------------------------------
            # CHECK EMPTY VALUES
            # ------------------------------------------------

            if not all([
                sepal_length_text,
                sepal_width_text,
                petal_length_text,
                petal_width_text
            ]):

                raise ValueError(
                    "All four measurements are required."
                )


            # ------------------------------------------------
            # CONVERT TO FLOAT
            # ------------------------------------------------

            sepal_length = float(
                sepal_length_text
            )

            sepal_width = float(
                sepal_width_text
            )

            petal_length = float(
                petal_length_text
            )

            petal_width = float(
                petal_width_text
            )


            # ------------------------------------------------
            # CHECK VALID NUMBERS
            # ------------------------------------------------

            measurements = [

                sepal_length,

                sepal_width,

                petal_length,

                petal_width

            ]


            if not all(
                math.isfinite(value)
                for value in measurements
            ):

                raise ValueError(
                    "Measurements must be valid numbers."
                )


            # ------------------------------------------------
            # CHECK POSITIVE VALUES
            # ------------------------------------------------

            if any(
                value <= 0
                for value in measurements
            ):

                raise ValueError(
                    "Measurements must be greater than zero."
                )


            # ------------------------------------------------
            # SAVE VALUES FOR HTML
            # ------------------------------------------------

            values = {

                "sepal_length": sepal_length,

                "sepal_width": sepal_width,

                "petal_length": petal_length,

                "petal_width": petal_width

            }


            # =================================================
            # CHECK MODEL
            # =================================================

            if model is None:

                raise RuntimeError(
                    "The machine-learning model could not be loaded."
                )


            # =================================================
            # PREPARE INPUT
            # =================================================

            input_data = [[

                sepal_length,

                sepal_width,

                petal_length,

                petal_width

            ]]


            # =================================================
            # MAKE PREDICTION
            # =================================================

            raw_prediction = model.predict(
                input_data
            )[0]


            # Convert prediction to standard name
            species = get_species_name(
                raw_prediction
            )


            # =================================================
            # SPECIES INFORMATION
            # =================================================

            info = flower_info.get(
                species,
                {

                    "display_name": species,

                    "scientific_name": species,

                    "symbol": "🌸",

                    "description":
                        "The model predicted this Iris species "
                        "based on the supplied flower measurements."

                }
            )


            # =================================================
            # PROBABILITIES
            # =================================================

            probabilities = {}


            if hasattr(model, "predict_proba"):

                probability_values = model.predict_proba(
                    input_data
                )[0]


                # Get class labels from the model
                model_classes = getattr(
                    model,
                    "classes_",
                    []
                )


                for class_value, probability in zip(
                    model_classes,
                    probability_values
                ):

                    class_name = standardize_class_name(
                        class_value
                    )


                    probabilities[class_name] = round(
                        float(probability) * 100,
                        2
                    )


                # Confidence is highest probability
                confidence = round(
                    max(probability_values) * 100,
                    2
                )

            else:

                confidence = None


            # =================================================
            # RESULT OBJECT
            #
            # This is what index.html uses:
            #
            # result.prediction
            # result.display_name
            # result.scientific_name
            # result.symbol
            # result.description
            # result.confidence
            # result.probabilities
            # =================================================

            result = {

                "prediction": species,

                "display_name":
                    info["display_name"],

                "scientific_name":
                    info["scientific_name"],

                "symbol":
                    info["symbol"],

                "description":
                    info["description"],

                "confidence":
                    confidence,

                "probabilities":
                    probabilities

            }


        # ====================================================
        # INPUT ERROR
        # ====================================================

        except ValueError:

            error = (
                "Please enter valid positive numerical "
                "values for all four measurements."
            )


        # ====================================================
        # OTHER ERROR
        # ====================================================

        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            error = (
                "Something went wrong while making "
                "the prediction. Please check your "
                "input values and try again."
            )


    # ========================================================
    # SEND DATA TO HTML
    # ========================================================

    return render_template(

        "index.html",

        result=result,

        values=values,

        error=error,

        model_ready=MODEL_READY

    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )