# Projet IAR 
Groupe : Ines Rahaoui, Inés Tian Ruiz-Bravo Plovins

Encadrant : Mehdi Khamassi

## Contexte 
Ce projet s’inscrit dans la continuité de l’article de Chartouny et al. (2024), qui propose un paradigme original pour étudier ces affordances à l’aide de l’apprentissage par renforcement model-based.

L’objectif principal de ce travail est double :
- reproduire certaines expériences clés de l’article, en particulier les résultats présentés dans la figure 6 ;
- proposer une extension exploratoire du modèle afin de tester ses limites, nous avons choisit d'explorer la théorie de l’esprit via des états latents.

## 1. Travail de reproduction
Les graphes à reproduire se trouvent dans le répertoire [imgs_report](imgs_report)

## 2. Extension : états sociaux latents et belief
Nous avons implémenté un nouvel agent `BeliefMB`, extension de l'agent `Basic_MB`, que vous pouvez trouver dans le fichier `script/agent.py`.

De plus, pour faire les simulations de la Social tasks nous avons aussi définit l'environnement `Lab_env_HRI_LatentObs` qui est très similaire à l'environnement `Lab_env_HRI` mais tenant en compte les observations des actions humaines :
- déplacement de l’humain,
- rotation de l’orientation de l’humain,
- variation de la distance humain–robot.

Ces modifications vous les trouverez dans le fichier `script/env.py`.

Finalement, un test préliminaire a été réaliser dans le fichier `script/test_belief_MB.py` pour analyser les résultats et les comparer avec un agent MB basic. 