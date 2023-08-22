import pandas as pd
import matplotlib.pyplot as plt

# Erreur de CSV?
try:
    data = pd.read_csv('data/data.csv')
except FileNotFoundError:
    print("The file data.csv was not found!")
    exit()
except pd.errors.EmptyDataError:
    print("The file data.csv is empty!")
    exit()

if data.empty or 'km' not in data.columns or 'price' not in data.columns:
    print("The file is empty or does not contain the required columns (km, price)!")
    exit()
try:
    data['km'] = pd.to_numeric(data['km'])
    data['price'] = pd.to_numeric(data['price'])
except ValueError:
    print("Error: The data contains non-numeric values!")
    exit()

if len(data) == 0:
    print("The CSV file is empty!")
    exit()

kilometrage = data['km'].values
prices = data['price'].values
len_prices = len(prices)

# Calcul de la moyenne des prix
mean_prices = sum(prices) / len_prices

# Calcul de la somme totale des carrés
sst = sum((price - mean_prices)**2 for price in prices)

# Data normalisation
# moyenne
kilometrage_mean = sum(kilometrage) / len(kilometrage)
# ecart type:
# **2 carrer, **0.5 racine carrer
kilometrage_std = (sum((mi - kilometrage_mean) ** 2 for mi in kilometrage) / len(kilometrage)) ** 0.5
# normalisation
kilometrage_normalized = [(mi - kilometrage_mean) / kilometrage_std for mi in kilometrage]


cost_history = []
theta0 = 0
theta1 = 0
learning_rate = 0.005
iterations = 2000

def estimate_price(kilometrage):
    return theta0 + theta1 * kilometrage

previous_cost = float('inf')

# Model de gradient
# L'entraînement du modèle utilise la descente de gradient pour trouver les valeurs optimales de theta0 et theta1
for y in range(iterations):
    sum_errors = 0
    sum_errors_kilometrage = 0
    # difference entre valeur prédite et valeur reel == errors
    for i in range(len_prices):
        sum_errors += (estimate_price(kilometrage_normalized[i]) - prices[i])
        sum_errors_kilometrage += (estimate_price(kilometrage_normalized[i]) - prices[i]) * kilometrage_normalized[i]
    
    temp_theta0 = theta0 - learning_rate * (1/len_prices) * sum_errors
    temp_theta1 = theta1 - learning_rate * (1/len_prices) * sum_errors_kilometrage
    
    theta0 = temp_theta0
    theta1 = temp_theta1
    
    # calcule de cout ca mesure à quel point le modèle s'adapte aux données.
    cost = sum([(estimate_price(kilometrage_normalized[i]) - prices[i])**2 for i in range(len_prices)]) / (2*len_prices)
    cost_history.append(cost)
    if cost >= previous_cost:
        # problème avec le taux d'apprentissage
        print("Gradient descent does not converge!")
        exit()
    previous_cost = cost

# Calcul de la somme des carrés des erreurs
sse = sum((estimate_price(kilometrage_normalized[i]) - prices[i])**2 for i in range(len_prices))

# Calcul du coefficient de détermination R^2
r2_score = 1 - (sse / sst)
r2_score = 100 * r2_score
print(f"La precision de l'algorithme est de:  {r2_score}")

with open('model_params/parameters.txt', 'w') as f:
    f.write(f"{theta0}\n")
    f.write(f"{theta1}\n")
    f.write(f"{kilometrage_mean}\n")
    f.write(f"{kilometrage_std}\n")

# Figure pour les trois subplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(kilometrage, prices, color='blue', label='Data points')
plt.xlabel('Kilometrage')
plt.ylabel('Price')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(kilometrage, prices, color='blue', label='Data points')
plt.plot(kilometrage, [estimate_price((x - kilometrage_mean) / kilometrage_std) for x in kilometrage], color='red', label='Regression line')
plt.xlabel('Kilometrage')
plt.ylabel('Price')
plt.text(0.05, 0.95, f'Précision: {r2_score:.2f}%', transform=plt.gca().transAxes, verticalalignment='top')
plt.legend()


# Subplot 3: Courbe d'apprentissage
plt.subplot(1, 3, 3)
plt.plot(cost_history)
plt.title('Courbe d\'apprentissage')
plt.xlabel('Itérations')
plt.ylabel('Coût')

# Afficher la figure avec les trois subplots
plt.show()
