## 3 Métholologies

#### Comparaison entre les modèles 

Pour tester les performances, nous avons prévu de comparer 3 modèles avec différents réglages :

- ArtClassifier (basé sur ResNet50 pré-entraîné)
- CustomArtCNN (réseau de neurones convolutif construit par nos soins, sans pré-entraînement)
- ArtClassifierTinyBased (basé sur ConvNeXt Tiny pré-entraîné)

Avec différents réglages :
- Epochs (10 ou 20 maximum)
- Early Stop (patience de 2 époques ou désactivé)
- Chargement des données (simple ou canonique)

Ce qui fait 2^3 combinaisons de réglages possibles.

Nous nous sommes réparties les tâches pour faire fonctionner les différents modèles sur nos machines. 

Le chargement simple conserve les catégories du dataset. Le chargement canonique regroupe plusieurs catégories dans des mouvements artistiques plus larges et traite les doublons repérés par leur nom de fichier.

Ce qui fait 2³ = 8 combinaisons de réglages possibles par modèle, soit 24 configurations pour les 3 modèles.

Le paramètre AVERAGE_ON est fixé à 2. Chaque configuration est donc exécutée deux fois afin de calculer une moyenne des résultats et un écart-type. Cela permet de voir si les performances varient entre les deux entraînements. Certains ont eu la possibilité d'en faire 2 d'autres non. La campagne complète représente donc 48 entraînements.


#### Mesure des performances
Pour automatiser les tests, nous avons utilisé le script measures.py, qui lance les différentes configurations et sauvegarde les résultats.

Les données sont séparées en deux parties : environ 80 % pour l’entraînement et 20 % pour la validation, avec une répartition par catégorie. Les performances sont mesurées sur les données de validation à la fin de chaque époque.

Nous avons utilisé plusieurs mesures pour comparer les modèles :
- Micro Top-1 : le pourcentage d’images pour lesquelles la première prédiction du modèle est correcte.
- Macro Top-1 : la moyenne des taux de réussite par catégorie. Chaque catégorie a le même poids, même si elle contient moins d’images.
- Top-5 : le pourcentage d’images pour lesquelles la bonne catégorie se trouve parmi les 5 premières prédictions.
- Loss de validation : la valeur de la fonction de perte sur les données de validation. Une valeur plus faible indique une meilleure performance selon cette fonction.

Lorsque l’Early Stop est activé, l’entraînement s’arrête après 2 époques consécutives sans amélioration suffisante de la loss de validation. Cela permet d’éviter de continuer un entraînement qui ne progresse plus.

Pour chaque configuration, nous conservons les meilleures valeurs de chaque mesure ainsi que celles de la dernière époque. Les résultats sont sauvegardés dans des fichiers JSON pour faciliter leur comparaison.

Ces mesures sont réalisées sur le jeu de validation ; le script ne prévoit pas de jeu de test indépendant.
