def load_parameters():
    try:
        with open('model_params/parameters.txt', 'r') as f:
            theta0 = float(f.readline())
            theta1 = float(f.readline())
            kilometrage_mean = float(f.readline())
            kilometrage_std = float(f.readline())
        return theta0, theta1, kilometrage_mean, kilometrage_std
    except FileNotFoundError:
        print("Erreur: Le fichier de paramètres du modèle est introuvable!")
        exit()
    except ValueError:
        print("Erreur: Les données dans le fichier de paramètres sont incorrectes!")
        exit()

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def main():
    # Charger les paramètres
    theta0, theta1, kilometrage_mean, kilometrage_std = load_parameters()

    try:
        mileage = float(input("Entrez le kilométrage de la voiture: "))
    except ValueError:
        print("Erreur: Veuillez entrer une valeur numérique pour le kilométrage!")
        return

    # Normaliser le kilométrage
    mileage_normalized = (mileage - kilometrage_mean) / kilometrage_std

    # Prédire le prix en utilisant le modèle entraîné
    predicted_price = estimate_price(mileage_normalized, theta0, theta1)

    if (predicted_price < 0):
        predicted_price = 0

    print(f"Le prix estimé pour une voiture avec {mileage} km est d'environ {predicted_price:.2f}€.")

if __name__ == "__main__":
    main()
