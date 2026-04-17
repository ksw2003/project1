import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

csv_path = "./data/sanbul2district-divby100.csv"
fires = pd.read_csv(csv_path)

print("===== Data Loaded =====")
print(fires.head())
print()

fires["longitude"] = fires["longitude"].astype(str).str.strip()
fires["latitude"] = fires["latitude"].astype(str).str.strip()
fires["month"] = fires["month"].astype(str).str.strip()
fires["day"] = fires["day"].astype(str).str.strip()

print("===== Data Types After Conversion =====")
print(fires.dtypes)
print()

print("===== Unique Values =====")
print("longitude:", sorted(fires["longitude"].unique()))
print("latitude :", sorted(fires["latitude"].unique()))
print("month    :", sorted(fires["month"].unique()))
print("day      :", sorted(fires["day"].unique()))
print()

fires_plot_before_log = fires.copy()
fires_plot_before_log["longitude_num"] = fires_plot_before_log["longitude"].astype(int)
fires_plot_before_log["latitude_num"] = fires_plot_before_log["latitude"].astype(int)

cols_for_hist = [
    "avg_temp",
    "avg_wind",
    "burned_area",
    "latitude_num",
    "longitude_num",
    "max_temp",
    "max_wind_speed"
]

fires_plot_before_log[cols_for_hist].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histogram of Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(fires["burned_area"], bins=30)
plt.title("burned_area")

plt.subplot(1, 2, 2)
plt.hist(np.log(fires["burned_area"] + 1), bins=30)
plt.title("ln(burned_area + 1)")

plt.tight_layout()
plt.show()
fires["burned_area"] = np.log(fires["burned_area"] + 1)

print("===== burned_area transformed with log(area + 1) =====")
print(fires["burned_area"].head())
print()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(fires, fires["month"]):
    train_set = fires.iloc[train_idx].copy()
    test_set = fires.iloc[test_idx].copy()

print("===== Train/Test Shape =====")
print("train_set:", train_set.shape)
print("test_set :", test_set.shape)
print()

fires["month"].hist(figsize=(8, 4))
plt.title("Month Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

scatter_cols = ["burned_area", "max_temp", "avg_temp", "max_wind_speed", "avg_wind"]
scatter_matrix(fires[scatter_cols], figsize=(10, 8))
plt.show()

fires_plot = fires.copy()
fires_plot["longitude_num"] = fires_plot["longitude"].astype(int)
fires_plot["latitude_num"] = fires_plot["latitude"].astype(int)

fires_plot.plot(
    kind="scatter",
    x="longitude_num",
    y="latitude_num",
    alpha=0.4,
    s=fires_plot["max_temp"] * 5,
    label="max_temp",
    c="burned_area",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    figsize=(8, 6)
)
plt.legend()
plt.title("Burned Area by Region")
plt.show()

fires_train = train_set.drop("burned_area", axis=1)
fires_labels = train_set["burned_area"].copy()

fires_test = test_set.drop("burned_area", axis=1)
fires_test_labels = test_set["burned_area"].copy()

print("\n############################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

num_attribs = ["avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["longitude", "latitude", "month", "day"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires_train)
fires_test_prepared = full_pipeline.transform(fires_test)

if hasattr(fires_prepared, "toarray"):
    fires_prepared = fires_prepared.toarray()

if hasattr(fires_test_prepared, "toarray"):
    fires_test_prepared = fires_test_prepared.toarray()

print("===== Prepared Data Shape =====")
print("fires_prepared      :", fires_prepared.shape)
print("fires_test_prepared :", fires_test_prepared.shape)
print()

X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared,
    fires_labels,
    test_size=0.2,
    random_state=42
)

X_test, y_test = fires_test_prepared, fires_test_labels

print("===== Train/Valid/Test Shape =====")
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_test :", X_test.shape)
print()

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_data=(X_valid, y_valid)
)

model.save('fires_model.keras')
joblib.dump(full_pipeline, 'fires_pipeline.pkl')

print("\n===== Saved Files =====")
print("fires_model.keras")
print("fires_pipeline.pkl")

test_loss = model.evaluate(X_test, y_test, verbose=0)
print("\n===== Test Evaluation =====")
print("Test Loss (MSE):", test_loss)

X_new = X_test[:3]
print("\n예측값(log scale):\n", np.round(model.predict(X_new), 2))

pred_log = model.predict(X_new).flatten()
pred_area = np.exp(pred_log) - 1
pred_area = np.maximum(pred_area, 0)

true_log = y_test.iloc[:3].values
true_area = np.exp(true_log) - 1

print("\n복원된 예측값(m²):", np.round(pred_area, 2))
print("실제값(m²):", np.round(true_area, 2))

history_df = pd.DataFrame(history.history)

plt.figure(figsize=(8, 5))
plt.plot(history_df.index + 1, history_df["loss"], label="train_loss")
plt.plot(history_df.index + 1, history_df["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()