import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pandas.errors

# Vérification de l'existence du fichier CSV et de sa non-vacuité
try:
    data = pd.read_csv('data/data.csv')
except FileNotFoundError:
    print("Le fichier data.csv n'a pas été trouvé!")
    exit()
except pandas.errors.EmptyDataError:
    print("Le fichier data.csv est vide!")
    exit()


# Vérification si le fichier est vide ou ne contient pas les colonnes attendues
if data.empty or 'km' not in data.columns or 'price' not in data.columns:
    print("Le fichier est vide ou ne contient pas les colonnes requises (km, price)!")
    exit()
try:
    data['km'] = pd.to_numeric(data['km'])
    data['price'] = pd.to_numeric(data['price'])
except ValueError:
    print("Erreur: Les données contiennent des valeurs non numériques!")
    exit()


# Filtrage des valeurs aberrantes
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data_filtered = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

if len(data) == 0:
    print("Le fichier CSV est vide!")
    exit()

X = data_filtered['km'].values
y = data_filtered['price'].values

# Normalisation des données
X_mean = sum(X) / len(X)
X_std = (sum((xi - X_mean) ** 2 for xi in X) / len(X)) ** 0.5
X_normalized = [(xi - X_mean) / X_std for xi in X]

# Initialisation des paramètres
theta0 = 0
theta1 = 0
learning_rate = 0.001
iterations = 10000
m = len(y)

# Fonction d'hypothèse
def estimate_price(mileage):
    return theta0 + theta1 * mileage

# Vérification de la convergence
prev_cost = float('inf')

# Entraînement du modèle avec descente de gradient
for _ in range(iterations):
    sum_errors = 0
    sum_errors_x = 0
    for i in range(m):
        sum_errors += (estimate_price(X_normalized[i]) - y[i])
        sum_errors_x += (estimate_price(X_normalized[i]) - y[i]) * X_normalized[i]
    
    temp_theta0 = theta0 - learning_rate * (1/m) * sum_errors
    temp_theta1 = theta1 - learning_rate * (1/m) * sum_errors_x
    
    theta0 = temp_theta0
    theta1 = temp_theta1
    
    # Calcul de la fonction de coût
    cost = sum([(estimate_price(X_normalized[i]) - y[i])**2 for i in range(m)]) / (2*m)
    if cost >= prev_cost:
        print("La descente de gradient ne converge pas!")
        exit()
    prev_cost = cost

# Sauvegarder les paramètres
with open('model_params/parameters.txt', 'w') as f:
    f.write(f"{theta0}\n")
    f.write(f"{theta1}\n")
    f.write(f"{X_mean}\n")
    f.write(f"{X_std}\n")

# Visualisation des données et de la ligne de régression
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, [estimate_price((x - X_mean) / X_std) for x in X], color='red', label='Regression line')
plt.xlabel('Kilométrage')
plt.ylabel('Prix')
plt.legend()
plt.show()
