import numpy as np
import csv
import math

def knn(chemin_fichier, nouveau_point, k=3):
    
    donnees = []
    etiquettes = []
    with open(chemin_fichier, 'r') as fichier:#ouvre le fichier de données
        lignes = fichier.readlines()# ignore les en-têtes   
        for ligne in lignes[1:]: # Pour chaque lignes (points)
            if ligne.strip():  # ignore les lignes vides
                elements = ligne.strip().split(',')  #  ajoute les 7 caratéristiques à la liste elements
                donnees.append([float(x) for x in elements[1:8]]) # ajoute les 7 caratéristiques convertit en float à la liste donnees       
                etiquettes.append(int(elements[-1])) #ajoute le label du point à la liste étiquettes
    donnees = np.array(donnees) 
    etiquettes = np.array(etiquettes) 
    # Calcule les distances entre le nouveau point et tous les points données
    distances = [] # Créer la liste vide qui contiendra les distances
    for point in donnees: # Parcours chaque point données
        distance = math.sqrt(sum((point[i] - nouveau_point[i]) ** 2 for i in range(len(point)))) # Calcule la distance euclidienne
        distances.append(distance) # Ajoute la distance calculée dans la liste
        
    # Trouve les k plus proches voisins
    indices_voisins = sorted(range(len(distances)), key=lambda i: distances[i])[:k] # Trie la liste des indices dans l'ordre décroissant des distances et prends les k premiers
    etiquettes_voisins = [etiquettes[i] for i in indices_voisins] #Ajoute labels des k plus proche voisins dans atiquettes voisin
     
    # Détermine la classe qui revient le plus souvent
    count = {}
    for etiquette in etiquettes_voisins: 
    # si l'étiquette est déja dans count ajoute 1 à sa clef , sinon on creer le couple clef valeurs etiquette:1
        if etiquette in count:
            count[etiquette] += 1
        else:
            count[etiquette] = 1
    classe_predite = max(count, key=count.get) #on garde le l'etiquette qui renviens leplus souvent
    
    return classe_predite #retourne la classe prédite



def KNN(entrainement, teste, predictions, k=3):
    
    resultats = []

    with open(teste, 'r') as fichier_test:
        lignes_test = fichier_test.readlines()
        en_tete = lignes_test[0].strip().split(',')
        id_index = en_tete.index('Id')  # Utilise "Id" pour les identifiants

        for ligne in lignes_test[1:]:
            if ligne.strip():  
                elements = ligne.strip().split(',')      
                identifiant = elements[id_index] # Affecte l'Id de l'élément à la variable identifiant
                caracteristiques = [float(x) for x in elements[1:8]]# Prends les caractéristiques de x   
                etiquette_predite = knn(entrainement, caracteristiques, k)# Prédit le label du point
                resultats.append((identifiant, etiquette_predite)) # Ajoute l'id et le label prédit dans resultats
    
    # Sauvegarde le fichier CSV contenant les étiquette prédites
    with open(predictions, 'w', newline='') as fichier_sortie:
        writer = csv.writer(fichier_sortie)
        writer.writerow(['Id', 'Label'])# Creer les en-têtes id et label
        writer.writerows(resultats)# Entre les résultats dans resultats



# Test
entrainement = "C:/Users/Tehin/Documents/ESILV/Data IA/iris/train.csv"
teste = "C:/Users/Tehin/Documents/ESILV/Data IA/iris/test.csv"
predictions = "C:/Users/Tehin/Documents/ESILV/Data IA/iris/Predictions.csv"

KNN(entrainement, teste, predictions)



