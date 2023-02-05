# Opérations de maintenances conditionnelles dans un système industriel à grande échelle

## Lancer le code
### Les agents non deep

Utiliser la commande `python run.py` et spécifier l'agent à lancer sur l'environnement :
Les différents agents sont répertoriés dans implemented_agents

Ex :
```
python run.py --agent=sarsa
```
Lancera l'agent sarsa sur l'environnement

### Les agents deep

Ils ne sont pas très fonctionnels sur l'environnement, 
- Le premier se trouve dans `run_deep_from_scratch.py` et se lance en en lançant son code. Il s'agit d'un DQN sur l'environnement discret qui travaille sur les états qui sont représentés en one-hot et renvoie la Q-Value estimé de chaque action.

- Le deuxième se lance grâce au fichier `run_deep.py` qui peut modifier directement l'architecture de l'agent.


## Structure du code

### L'environnement
On s'est inspiré des environnements utilisant gym, notre environnement est très modulable et se trouve dans le dossier env

#### Les Actions

Les actions sont composées d'un nombre de réparation préventive et d'un nombre de réparation corrective. Ce nombre de réparation et ensuite traité par le fichier environnement et les réparations sont réparties sur les différentes éoliennes.

#### Les Etats 

Les Etats sont représentés par une liste d'éoliennes (item), chaque éolienne est caractérisée par son état de dégradation (wear) et par d'autres variables de dégradations. Dans notre cas d'études ces autres variables, tel que le treshold à partir duquel l'éolienne est H.S. ou la fonction de dégradation de l'éolienne, sont fixes et ne caractérisent donc pas les item mais l'environnement.

### Les différentes éxécutions

Les différentes éxécutions possibles peuvent être sélectionnées en modifiant le string à l'origine de création de l'environnement `env = Environnement.init("3dadvanced")`, dans ce string, on peut choisir le nombre d'éoliennes : le premier chiffre, si on est en discret ou continu : "c" pour continu ou "d" pour discret et enfin, on peut utiliser une des 2 fonction de dégradation implémentées. Si on laisse "5d" par exemple, on utilisera la fonction de dégradation de base en discret avec 5 éoliennes. EN mettant "advanced" on choisit la fonction de dégradation plus réaliste avec une dégradation plus lente

Ces fonctions sont implémentées dans `wearing_functions.py` et des nouvelles peuvent être ajoutées dedans en les reliant à un mot dans `execution.py`

### L'environnement 

Il contient toutes les fonctions nécessaires pour lancer un agent de RL :
- reset : à appeler avant de lancer l'agent où pour repartir de l'état initial
- step : pour, avec une action, passer d'un état à un autre. L'état est une propriété de l'environnement, si on veut se déplacer d'états en états dans un sens autre qu'en fonction du temps, il faudra faire des copies de l'objet environnement.
- render : permet d'enrigstrer une moyenne des reward pendant une éxécution normale. On utilise wandb pour visualiser plus facilement les courbes. 


## Les différents agents

Le fichier agent contient plusieurs implémentation d'agents, `agent.py` donne l'architecture que doit avoir un agent pour pouvoir être lancé par la fonction `run.py`. 

### Agent réflexe
L'agent réflexe est le premier agent baseline qu'on a implémenté pour pouvoir comparer les résultats des agents de RL. Cet agent est implémenté en essayant de copier le comportement d'un humain prenant des décisions. L'objectif premier était donc de dépasser cet agent là. Il prend des actions quand une éolienne est très dégradée, pour la réparée.

### Agents de RL (non deep)
3 agents de RL non deep on été implémenté, afin de comparer leurs performances sur l'environnement au début du projet.
- L'agent SARSA est un agent online, qui se comporte avec une politique epsilon greedy et apprend avec ses mouvements donc la même politique
- L'agent SARSA expected est un agent offline, qui se comporte avec la politique epsilon greedy et apprend avec cette même politique
- L'agent QLearning est un agent offline, qui se comporte avec la politique epsilon greedy et apprend avec la politique greedy

L'agent de QLearning est le plus performant dans notre cas d'étude 

### Solver

L'agent solver marche à base de *Markov Decision Process*, il se trouve dans le fichier solver_mdp. On peut en extraire sa politique (liste des actions optimales pour chaque états). Il marche aussi bien pour un délais de prise en compte des actions fixé.


## Références

Le `Rapport Final` se trouve dans le dossier Documentation et explique plus en détail le code et les conclusions du projet

[1] Agent DQN en pytorch : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

[2] https://automatants.cs-campus.fr/formations : Slides de la formation “Reinforcement
Learning – DQN” de 2021-2022

[3] TP dans le cadre du camp “ML4Good” de EffiSciences : https://www.effisciences.org/

[4] : Reinforcement Learning Course by David Silver (deepmind)
https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=1