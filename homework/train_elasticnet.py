# importacion de librerias
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# descarga de datos
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# preparacion de datos
y = df["quality"]
x = df.copy()
x.pop("quality")

# dividir los datos en entrenamiento y testing
(x_train, x_test, y_train, y_test) = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=123456,
)

# entrenar el modelo
estimator = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=12345)
estimator.fit(x_train, y_train)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
y_pred = estimator.predict(x_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print()
print("Metricas de entrenamiento:")
print(f"  MSE: {mse}")
print(f"  MAE: {mae}")
print(f"  R2: {r2}")

# Metricas de error durante testing
print()
print("Metricas de testing:")
y_pred = estimator.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  MSE: {mse}")
print(f"  MAE: {mae}")
print(f"  R2: {r2}")
