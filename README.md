# Projet7API

Pour l’API nous aurons à notre disposition 3 urls différentes, celle de base qui sera 127.0.0.1:5000 en local après avoir lancé le script ou https://projetpythonapi.herokuapp.com/ sur le cloud.

Lorsque vous arriverez sur la page d'acceuil, les ID disponibles seront affichées.

Ensuite nous pourrons ajouter:

	1) /infoClient?id= (int id client)
	
	2) /predictionClient?id= (int id client)
	
	3) /predictionModeleClient?id= (int id client)

Le 1 va nous renseigner sur le client en nous donnant quelques renseignements, par exemple nous aurons le nombre d’enfant, les revenus ou le prêt.

Le 2 va nous renseigner si le client va avoir son crédit ou non, en utilisant les scores enregistrés au préalable  en ligne car pour cette approche nous avons besoin d’un fichier qui est trop lourd pour être déposé en ligne.

Le 3 utilisera le meilleur modèle pour prédire le résultat de chaque client.
