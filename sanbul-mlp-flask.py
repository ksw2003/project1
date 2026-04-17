import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# 반복 줄이기용 별칭
STRING_FIELD = StringField

np.random.seed(42)

app = Flask(__name__)
app.config["SECRET_KEY"] = "hard to guess string"
bootstrap = Bootstrap5(app)

model = keras.models.load_model("fires_model.keras")
full_pipeline = joblib.load("fires_pipeline.pkl")


class LabForm(FlaskForm):
    longitude = STRING_FIELD("longitude(1~7)", validators=[DataRequired()])
    latitude = STRING_FIELD("latitude(1~7)", validators=[DataRequired()])
    month = STRING_FIELD("month(01-Jan ~ Dec-12)", validators=[DataRequired()])
    day = STRING_FIELD("day(00-sun ~ 06-sat, 07-hol)", validators=[DataRequired()])
    avg_temp = STRING_FIELD("avg_temp", validators=[DataRequired()])
    max_temp = STRING_FIELD("max_temp", validators=[DataRequired()])
    max_wind_speed = STRING_FIELD("max_wind_speed", validators=[DataRequired()])
    avg_wind = STRING_FIELD("avg_wind", validators=[DataRequired()])
    submit = SubmitField("Submit")


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def lab():
    form = LabForm()

    if form.validate_on_submit():
        longitude_value = form.longitude.data.strip()
        latitude_value = form.latitude.data.strip()
        month_value = form.month.data.strip()
        day_value = form.day.data.strip()

        if "-" in month_value:
            left, right = month_value.split("-", 1)
            month_value = f"{left}-{right.capitalize()}"

        if "-" in day_value:
            left, right = day_value.split("-", 1)
            day_value = f"{left}-{right.lower()}"

        user_dict = {
            "longitude": [str(longitude_value)],
            "latitude": [str(latitude_value)],
            "month": [str(month_value)],
            "day": [str(day_value)],
            "avg_temp": [float(form.avg_temp.data)],
            "max_temp": [float(form.max_temp.data)],
            "max_wind_speed": [float(form.max_wind_speed.data)],
            "avg_wind": [float(form.avg_wind.data)],
        }

        user_df = pd.DataFrame(user_dict)

        print("\n===== USER INPUT DATAFRAME =====")
        print(user_df)
        print("\n===== USER INPUT DTYPES =====")
        print(user_df.dtypes)

        X_user = full_pipeline.transform(user_df)

        print("\n===== TRANSFORMED SHAPE =====")
        print(X_user.shape)

        pred = model.predict(X_user, verbose=0)

        raw_log_pred = float(pred[0][0])
        pred_value = np.exp(raw_log_pred) - 1
        pred_value = max(0, pred_value)
        pred_value = np.round(pred_value, 2)

        print("raw_log_pred =", raw_log_pred)
        print("restored_area =", pred_value)

        return render_template("result.html", prediction=pred_value)

    return render_template("prediction.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)