#!/usr/bin/env python
# coding: utf-8

# # Analyse exploratoire et modèle prédictif - Etude de cas du Titanic

# ## Etape 1 - Data Understanding

# “Le transatlantique Titanic, le plus grand paquebot du monde, appartenant à la compagnie anglaise The White Star Line, a heurté la nuit dernière contre un iceberg, près des bancs de Terre-Neuve, et a coulé. Fort heureusement, les secours ont été prompts, et les passagers, au nombre de 2700, y compris l'équipage, ont pu être tous sauvés.”
# C’est ce que nous pouvions lire le 16 avril 1912 dans l’Echo de Paris, mais aurions-nous pu prédire qui allait survivre ? C’est ce à quoi nous allons tenter de répondre à travers cette analyse. 
# 
# Pour cela nous allons : 
# - Analyser et identifier les facteurs favorisant la survie 
# - Mettre en place un modèle de prédiction permettant de déterminer qui survivra sur nos données de test

# ### Import des libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install xgboost')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, make_scorer, confusion_matrix, f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer

# Importation des classifieurs Naive Bayes, régression logistique, Bagging, RandomForest, AdaBoost, GradientBoost, arbres de décision, SVM et XGBoost
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# On va définir ici le style de tracé (graphique)
plt.style.use('seaborn-notebook')
from matplotlib.ticker import StrMethodFormatter

sns.set(style='white', context='notebook', palette='deep')


# ### Lecture des données d'entraînement et de test

# In[2]:


# Ici nous allons chercher à nous connecter à nos données en indiquant notre répertoire
train = pd.read_csv("C:/Users/kcho/Desktop/Titanic/train.csv")
test = pd.read_csv("C:/Users/kcho/Desktop/Titanic/test.csv")
IDtest = test["PassengerId"]


# Pour réaliser cette étude de cas nous allons utiliser la méthode CRISP-DM c'est à dire : 

# ## Etape 2 : Data Understanding

# ### Description du dataset 'train'

# In[3]:


train.head()


# Cet aperçu nous permet de mieux comprendre les différentes composantes de notre dataset et d’amener nos premières pistes de réflexions. Voici un résumé pour chacune des colonnes : 
# 
# - PassengerId : Numéro d'identification unique d'un passager
# - Survival : le passager a survécu ou non ; 1 s'il a survécu et 0 s'il n'a pas survécu.
# - Pclass : Classe du ticket (1= première classe, 2= seconde classe,3= troisième classe). Peut être considéré comme un élément permettant d’évaluer le statut socio-économique de l’individu.
# - Sex : sexe
# - Age : âge en années
# - Sibsp : Nombre de frères et sœurs / conjoints à bord du Titanic
# - Parch : Nombre de parents / enfants à bord du Titanic
# - Ticket : Numéro du ticket
# - Fare : Tarif passager
# - Cabin : Numéro de la cabine
# - Embarked : Port d'embarquement (C = Cherbourg, Q = Queenstown, S = Southampton)
# 
# On peut les regrouper en fonction de la typologie de variables auxquelles elles appartiennent : 
# 
# - Variables catégorielles :
# 
#     - Nominales (variables ayant deux catégories ou plus, mais qui n'ont pas d'ordre intrinsèque)
#         Cabin
#         Embarked (Port d'embarquement : C (Cherbourg),Q (Queenstown), S (Southampton)
# 
#     - Dichotomiques (variable nominale avec seulement deux catégories)
#         Sex (Homme/Femme)
# 
# - Ordinales (variables ayant deux catégories ou plus, tout comme les variables nominales. Seules les catégories peuvent également être ordonnées ou classées.)
#     - Pclass (statut socio-économique : 1 (Première classe, classe premium), 2 (Seconde classe), 3 (Troisème classe)
# - Variables numériques :
#     - Discrètes
#         PassengerID(identifiant unique pour chaque passager)
#         SibSp
#         Parch
#         Survived (notre résultat ou variable dépendante : 0 ou 1)
# 
# - Continues
#     Age
#     Fare
# 
# - Variables textuelles :
#     Ticket (numéro de billet pour le passager)
#     Name (nom du passager)
# 

# In[4]:


train.info()


# In[5]:


train.shape


# In[6]:


train.describe()


# ### Description du dataset 'test'

# In[7]:


test.head()


# In[8]:


test.info()


# In[9]:


test.shape


# In[10]:


test.describe()


# Les tableaux ci-dessus nous permettent de constater certaines choses :
# - Nous avons quelques variables catégorielles qui doivent être converties en données numériques afin que les algorithmes d'apprentissage automatique puissent les traiter.
# - Les features ont des échelles très différentes et nous devrons les convertir à peu près à la même échelle.
# - Certaines features contiennent des valeurs manquantes (NaN = Not a Number), que nous devons traiter.

# ## Identification des outliners et traitement des valeurs manquantes

#  Nous allons donc tenter d'approfondir nos analyses concernant les données à notre disposition pour tenter d'identifier : 
#  - Les valeurs aberrantes/outliers
#  - Les doublons
#  - Les valeurs manquantes
#  
# Une fois cela fait il sera possible de déterminer comment les traiter.

# ### Les valeurs manquantes

# In[11]:


print (train.isnull().sum())
print (''.center(20, "*"))
print (test.isnull().sum())
sns.boxplot(x='Survived',y='Fare',data=train)
missing_values=train.isnull().sum() 
missing_values[missing_values>0]/len(train)*100


# Ici on s'aperçoit que les colonnes 'Age', 'Cabin' et 'Embarked' ont des valeurs manquantes. Pour le dataset de train on constate 19,87% de valeurs manquantes pour l'âge, 77,10% pour le numéro de la Cabine et 0,2% de valeurs manquantes pour le point d'embarquement.
# 
# Les données manquantes dans l'ensemble de données d'entraînement peuvent réduire l'ajustement d'un modèle ou conduire à un modèle biaisé car nous n'avons pas correctement analysé le comportement et la relation avec d'autres variables. Cela peut conduire à des prédictions ou classifications erronées. Pour éviter cette problématique il est nécessaire de traiter ces valeurs manquantes en prennant en compte le contexte.

# ### Identification des outliers pour les variables numériques

# J'ai voulu utiliser la méthode de Tukey pour identifier des outliers. Cependant la méthode de Tukey est opérationnelle lorsque nous sommes en proie à une distribution qui n'est pas normale. Nous allons donc commencer par afficher la distribution pour les variables numériques "Age", "SibSp", "Parch", et "Fare" afin de vérifier qu'elles ne répondent pas à une distribution normale.

# In[12]:


import matplotlib.pyplot as plt

for col in ["Age", "SibSp", "Parch", "Fare"]:
    plt.hist(train[col])
    plt.title(col)
    plt.show()


# Afficher les histogrammes pour les variables numériques peut nous permettre de déterminer la direction dans laquelle ces variables sont distribuées, les valeurs aberrantes apparaîtront en dehors de la distribution globale des données. Si l'histogramme est asymétrique à droite ou à gauche, cela indique la présence de valeurs extrêmes ou de valeurs aberrantes. Ici nous pouvons donc déduire que certaines valeurs sont aberrantes (excepté pour l'age). L'âge n'est pas un problème car même avec une valeur min à 0,17 on peut très bien en déduire qu'il s'agit d'un bébé. Pour traiter les valeurs aberrantes on peut réaliser la méthode de Tukey cependant il faut vérifier que la distribution n'est pas normale c'est le but de l'étape suivante.

# In[13]:


from scipy.stats import shapiro
import pandas as pd

# spécifier les colonnes à tester
cols_to_test = ["Survived", "PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]

# effectuer le test de Shapiro-Wilk sur chaque colonne
for col in cols_to_test:
    # extraire la colonne du dataframe
    data = train[col]

    # effectuer le test de Shapiro-Wilk sur la colonne
    stat, p = shapiro(data)

    # afficher le résultat du test
    print("Colonne : ", col)
    print("Statistiques du test de Shapiro-Wilk : ", stat)
    print("p-value : ", p)

    # interpréter le résultat du test
    alpha = 0.05
    if p > alpha:
        print("La distribution des données est normale (on ne peut pas rejeter H0)")
    else:
        print("La distribution des données n'est pas normale (on rejette H0)")


# On constate donc que les différentes distributions des variables que nous avions identifiés ne sont pas normales. Nous pouvons donc utiliser la méthode de Tukey pour identifier les outliers.

# In[14]:


# Détection des outliers

def detect_outliers(df,n,features):
    outlier_indices = []
    
    # Iterer sur chacune des colonnes du dataset
    for col in features:
        # Premier quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # Troisème quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Intervalle interquartile (le fameux IQR)
        IQR = Q3 - Q1
        
        # Niveau de la valeur aberrante que je fixe selon Turkey
        outlier_step = 1.5 * IQR
        
        # Déterminer une liste d'indices de valeurs aberrantes pour col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # Ajouter les indices de valeurs aberrantes trouvés pour col à la liste des indices de valeurs aberrantes
        outlier_indices.extend(outlier_list_col)
        
    # Sélectionner les observations contenant plus de 2 valeurs aberrantes
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# Détection des outliers des colonnes Age, SibSp , Parch and Fare : uniquement pour les valeurs numériques
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# In[15]:


train.loc[Outliers_to_drop] # On va afficher les outliers


# Après quelques recherches sur internet concernant les informations sur le prix des billets que l'on aurait pu croire comme aberrantes avec une valeur de 512 (valeur max) sont finalement plausibles car certains voyageurs prennaient des appartements en plus etc... Nous détectons tout de même 10 valeurs aberrantes. Les 28, 89 et 342 passagers ont un 'Fare' élevé par rapport aux quartiles. Les 7 autres ont des valeurs très élevées de SibSP. Nous pouvons donc supprimer ces outliers, c'est le rôle de la ligne suivante.

# In[16]:


# Suppression des outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# ### Identification des doublons

# Une fois les outliers identifiées nous pouvons passer aux doublons. La ligne de code ci-dessous va nous permettre de compter les enregistrements en double.

# In[17]:


# Trouver le nombre d'enregistrements en doublon
print('train - Nombre denregistrements en double:', train.duplicated().sum())
print('test - Nombre denregistrements en double:', test.duplicated().sum())


# Dans notre cas on constate l'absence de doublons ce qui est plutôt positif pour nous car aucun retraitement ne sera à effectuer.

# ### Traitement des valeurs manquantes 

# Une des étapes les plus délicates pour permettre à notre modèle d'avoir une bonne fiabilité ainsi que pour mener une analyse exploratoire digne de ce nom correspond au traitement des valeurs manquantes. Un mauvais traitement dans les valeurs manquantes (sans prise en compte du contexte) amènera à des données biaisées et à un manque de fiabilité de notre modèle. Nous allons faire en sorte de prendre en maximum le contexte des données pour déterminer comment traiter ces données manquantes.

# #### Traitement des valeurs manquantes pour l'Age 

# Nous avons dans les premières lignes de l'exploration de notre jeu de données constatés que l'age possèdaient de nombreuses lignes avec des valeurs manquantes. Il existe plusieurs manières de traiter ces lignes vides, la plus courante étant de calculer la moyenne d'âge et de l'imputer aux lignes manquantes. Cependant, on constate ici une diversité dans les profils de notre base de données. Si bien que j'ai décidé d'utiliser le titre présent dans le nom pour déterminer l'age de la ligne. Je m'explique, au lieu de calculer la moyenne sur toute la base de données je vais calculer la moyenne des âges des personnes possédants le même titre que la personne dont l'age est manquant. Ainsi nous gagnons en fiabilité, il est rare d'avoir un titre de 'Capt' lorsque l'on a 4 ans. Je suis donc parti du principe qu'il était de même pour les autres titres.

# In[18]:


# Comme le test n'a qu'une seule valeur manquante, remplissons-la avec la moyenne.
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

# Concatenation des données d'entrainement et de test pour le traitement
data_df = pd.concat([train, test], ignore_index=True)

# Extraction des titres à partir des noms des passagers
data_df['Title'] = data_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)

# Remplacement des titres rares par les plus courants
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
           'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)

# Affichage des différents titres et leur nombre d'occurrences
print(data_df['Title'].value_counts())

# Imputation de l'âge manquant en utilisant la moyenne de l'âge par titre
titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Rev', 'Dr']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].mean()[title]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

# Séparation des données d'entrainement et de test mises à jour
train = data_df.iloc[:len(train)]
test = data_df.iloc[len(train):]

# Vérification du nombre de valeurs nulles dans les données d'entrainement et de test
print(train.isnull().sum())
print(test.isnull().sum())


# Nous voilà avec un traitement des données manquantes pour l'âge : 

# In[19]:


train.head()


# #### Traitement des valeurs manquantes pour le numéro de Cabine et suppresion de la colonne 'Ticket'

# Au vue du nombre de données manquantes qui correspondent à 77% des données totales il est préférable de supprimer cette colonne. De plus, le numéro de cabine est difficilement exploitable en l'état car a une partie numérique et alphabétique. Par la même occasion nous allons supprimer la colonne 'Ticket' qui n'a aucune valeur dans notre analyse

# In[20]:


print("Avant", train.shape, test.shape)

train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)

("Après", train.shape, test.shape)


# #### Traitement du sexe en changeant la catégorie

# Afin de faciliter l'exploitation de nos données concernant le sexe ainsi que le traitement de notre modèle il est préférable de modifier le type de cette variable. De base, nous avions des chaines de caractère. Nous allons donc attribuer une valeur à chacun des 2 sexes (0 pour l'homme, 1 pour la femme).

# In[21]:


# Mapper chaque valeur de la variable "Sexe" à une valeur numérique
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# #### Traitement des valeurs du port d'embarquement 

# Pour traiter le faible nombre de valeurs manquantes pour le port d'embarquement j'ai décidé de remplacer les valeurs manquantes par la valeur la plus récurrente. La première ligne va nous permettre de déterminer la valeur la plus récurrente et la deuxième ligne complétera les valeurs manquantes du port d'embarquement en attribuant cette valeur.

# In[22]:


freq_port = train.Embarked.dropna().mode()[0]
freq_port


# In[23]:


for dataset in [train, test]:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Analyse des résultats et interprétation 

# Cette partie va nous permettre d'identifier des liens entre les variables et notre chance de survie. J'ai commencé par faire une heatmap permettant d'identifier les facteurs corrélés à notre survie. Cette heatmap servira de base à nos hypothèses et de pistes à explorer.

# In[24]:


heatmap = sns.heatmap(train.corr()[['Survived']].sort_values(by='Survived', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Survived', fontdict={'fontsize':18}, pad=16);


# ##### Hypothèse 1 : Combien de personnes ont embarqué depuis différents ports ? Y a-t-il une corrélation entre le port d'embarquement et la survie ?

# In[25]:


train['Embarked'].value_counts()/len(train)


# In[26]:


sns.set(style="darkgrid")
sns.countplot( x='Embarked', data=train, hue="Embarked", palette="Set1");


# On constate qu'il y a une forte proportion des voyageurs qui ont embarqués par le port de Southampton.

# In[27]:


sns.set(style="darkgrid")
sns.countplot( x='Survived', data=train, hue="Embarked", palette="Set1");


# In[28]:


train.groupby('Embarked').mean()


# Cependant on remarque que l'espérance de vie est meilleure pour les voyageurs ayant embarqués sur le port de Cherbourg. Après quelques recherches sur internet cette espérance peut être expliquée par un autre facteur : le niveau de vie. En effet, les habitants de Cherbourg ont un meilleur niveau de vie que les habitants des 2 autres ports, cela implique également un investissement plus important dans l'achat de billet et donc sûrement une classe plus importante. Cette hypothèse est également confirmée par le prix du billet que l'on constate à 59,95 vs 13 et 25.

# #####  Q2: Est-ce que la survie dépend du sexe ?

# In[29]:


train.groupby('Sex').mean()


# Le résultat est sans équivoque. Le cas du titanic est l'exemple parfait du dicton "les femmes et les enfants d'abord". Avec 75% de survie constatée chez les femmes le résultat est sans appel. Cependant, il est possible de vérifier si ces résultats ne peuvnet pas être croisés avec d'autres données.

# In[30]:


FacetGrid = sns.FacetGrid(train, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None )
FacetGrid.add_legend();


# Il semble y avoir une corrélation entre le port d'embarquement et la survie, selon le sexe des passagers. Les femmes qui ont embarqué au port Q et au port S ont une plus grande chance de survie. C'est le contraire si elles ont embarqué au port C. Les hommes ont une probabilité de survie élevée s'ils ont embarqué au port C, mais une faible probabilité s'ils ont embarqué au port Q ou S. Le niveau de la classe de voyage (Pclass) est également corrélé à la survie. 

# In[31]:


train[["Sex","Survived"]].groupby('Sex').mean()


# In[32]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
femme = train[train['Sex']==1]
homme = train[train['Sex']==0]
ax = sns.distplot(femme[femme['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")
ax = sns.distplot(femme[femme['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")
ax.legend()
ax.set_title('Femme')
ax = sns.distplot(homme[homme['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")
ax = sns.distplot(homme[homme['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")
ax.legend()
_ = ax.set_title('Homme');


# On peut voir que les hommes ont une forte probabilité de survie quand ils ont entre 18 et 30 ans, ce qui est également un peu vrai pour les femmes mais pas complètement. Pour les femmes, les chances de survie sont plus élevées entre 14 et 40 ans. Sachant que les femmes ont généralement une plus forte probabilité de survie.
# Pour les hommes, la probabilité de survie est très élevé pour les bébés/nourrissons et pour les jeunes hommes mais c'est également le cas pour les femmes. 

# #### Q3 : Est-ce que la classe détermine la probabilité de survie ? 

# In[33]:


sns.barplot(x='Pclass', y='Survived', data=train);


# In[34]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived
print("Pourcentage de la première classe qui ont survécu:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Pourcentage de la deuxième classe qui ont survécu", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Pourcentage de la troisième classe qui ont survécu", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# In[35]:


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
new_labels = ['Homme', 'Femme']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
g = g.set_ylabels("survival probability")


# In[36]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=3.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# Ici, nous constatons clairement que la classe (Pclass) a une influence sur les chances de survie d'une personne, en particulier si cette personne est en première classe (1). 
# Le graphique ci-dessus confirme notre hypothèse sur la classe 1, mais nous pouvons également constater une forte probabilité qu'une personne en classe 3 ne survivra pas.

# #### Q4 : Est-ce que le prix du billet détermine la probabilité de survivre ? 

# In[37]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# Nous constatons donc que oui, le prix du billet influence bien les chances de survie puisque les tarifs les plus élevés sont ceux qui ont survécu peu importe le port d'embarquement.

# #### Q5 - Est-ce que le titre donne une meilleure chance de survie ? 

# In[38]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# En effet, on constate que le titre a également son rôle a joué. Mais davantage car il est lié au sexe ou au niveau de vie de l'individu qui le porte.

# #### Q6 : Les passagers qui voyagent seuls ont-ils de meilleures chances de survie ?

# SibSp et Parch auraient plus de sens en tant que features combinée, qui montre le nombre total de parents qu'une personne a sur le Titanic. Je vais la créer ci-dessous ainsi qu'une fonctionnalité qui montre si quelqu'un est seul ou non.

# In[39]:


data = [train, test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'voyageur_seul'] = 'Non'
    dataset.loc[dataset['relatives'] == 0, 'voyageur_seul'] = 'Oui'
train['voyageur_seul'].value_counts()


# In[40]:


train['relatives'].value_counts()


# In[41]:


sns.barplot(x='voyageur_seul', y='Survived', data=train);


# In[42]:


sns.barplot(x='SibSp', y='Survived', data=train);


# In[43]:


sns.barplot(x='Parch', y='Survived', data=train);


# In[44]:


axes = sns.factorplot('relatives','Survived', 
                      data=train, aspect = 2.5, );


# On peut constater ici que l'on a une probabilité de survie élevée avec 1 à 3 personnes, mais une probabilité plus faible si on a moins de 1 ou plus de 3 personnes avec nous (sauf pour certains cas avec 6 personnes apparentées). Donc non, il était préférable pour vous de ne pas être seul lorsque le titanic a coulé.

# In[45]:


# Matrice de corrélation entre les valeurs numériques (SibSp, Parch, Age, Fare, Title, Pclass) et Survived.
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare","Pclass","voyageur_seul", "Title"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# L'avantage de cette matrice corrélation est qu'il est facile d'interpréter nos résultats et qu'elle permet de récapituler les différents constats que nous avons pu établir au préalable.
# A savoir l'impact de : 
# - La classe sur l'espérance de survie (une meilleure classe est égale à plus de chance de survie)
# - Le prix du billet sur l'espérance de survie (plus le billet est cher plus vous avez de chance de survivre)
# - Le sexe et l'âge (les enfants et les femmes ont davantage de chance de survivre)
# - Le nombre de personnes avec qui vous voyagez (être seul n'est pas favorable mais être trop nombreux non plus).

# ## Etape 3 : Data Preparation

# ### Feature engineering

# Maintenant que nous avons réalisé nos analyses nous allons devoir traiter certaines variables afin qu'elles soient interprétables pour notre modèle. C'est notamment le cas pour : 
# - Le port d'embarquement qu'il faut convertir en valeur numérique
# - Le titre qu'il faut également convertir en valeur numérique

# #### Conversion de la variable du port d'embarquement en attribuant des valeurs numériques

# In[46]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train, test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[47]:


# Vérification que la modification est bien appliquée pour le dataset de train
train.head()


# In[48]:


# Vérification que la modification est bien appliquée pour le dataset de test
test.head()


# #### Conversion de la variable du titre en attribuant des valeurs numériques

# In[49]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rev": 5, "Dr": 6}
data = [train, test]

for dataset in data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[50]:


# Vérification que la modification est bien appliquée pour le dataset de train
train.head()


# In[51]:


# Vérification que la modification est bien appliquée pour le dataset de test
test.head()


# ### Préparation du modèle de Machine learning

# Ces lignes permettent de préparer les données pour un modèle de machine learning en supprimant certaines colonnes, en extrayant la variable cible "Survived" et en remplaçant les valeurs manquantes dans le jeu de données de test.

# In[52]:


get_ipython().system('pip install catboost')


# In[53]:


X_train = train.drop(["Survived","Name","voyageur_seul"], axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId","Name","voyageur_seul"], axis=1).copy()
X_test["Survived"] = X_test["Survived"].fillna(-1)
X_train.shape, Y_train.shape, X_test.shape


# Une fois les données préparées il suffit de mettre en place le modèle désiré et de sélectionner le plus pertinent. J'ai décidé de fonctionner en deux étapes:
# 1) La première méthode ne divise pas les données en jeux de validation et d'entraînement. En effet, dans la première méthode le modèle se base sur l'accuracy (la précision) il est construit en utilisant l'ensemble des données d'entraînement (X_train et Y_train) sans diviser les données en ensembles de validation et d'entraînement distincts. 
# 
# 2) La deuxième méthode utilise la validation croisée avec StratifiedShuffleSplit, qui divise les données en plusieurs jeux de validation et d'entraînement pour évaluer la performance du modèle de manière plus fiable. Dans cette méthode je vais réaliser une validation croisée sur trois splits différents de mes données d'entraînement
# 
# Cela permettra d'avoir 2 points de vue différents sur nos modèles et de croiser les méthodologies.

# ## Etape 4 : Modeling

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#Pour ignorer les warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[55]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
Y_pred = gbk.predict(X_test)
acc_gbk = round(gbk.score(X_train, Y_train) * 100, 2)
acc_gbk


# In[56]:


# Support Vector Machines
#Pour ignorer les warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[57]:


# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[58]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[59]:


#Perceptron
perceptron = Perceptron()
knn = KNeighborsClassifier(n_neighbors = 3)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[60]:


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# Avec cette première méthodologie on obtient un modèle à 90,47% avec le 'Gradient Boosting Classifier'. Ce serait donc le modèle le plus performant sans split les données d'entrainement. Cependant, dans cette méthodologie nous avons un risque d'overfitting. Voyons les résultats de notre deuxième méthode.

# In[61]:


from sklearn.model_selection import train_test_split

# Supprimer les colonnes inutiles et la colonne "Survived" du jeu de test
X_train = train.drop(["Survived","Name","voyageur_seul"], axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId","Name","voyageur_seul"], axis=1).copy()

# Remplacer les valeurs manquantes de la colonne "Survived" par -1
X_test["Survived"] = X_test["Survived"].fillna(-1)

# Diviser les données d'entraînement en jeux d'entraînement et de validation
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.22, random_state = 0)

# Afficher les dimensions des jeux de données
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(X_test.shape)


# In[62]:


import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Supprimer les colonnes inutiles et la colonne "Survived" du jeu de test
X_train = train.drop(["Survived","Name","voyageur_seul"], axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId","Name","voyageur_seul"], axis=1).copy()

# Remplacer les valeurs manquantes de la colonne "Survived" par -1
X_test["Survived"] = X_test["Survived"].fillna(-1)

# Diviser les données d'entraînement en jeux d'entraînement et de validation
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.22, random_state = 0)

# Afficher les dimensions des jeux de données
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(X_test.shape)

# Définir les classifieurs à tester
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    CatBoostClassifier(),
    XGBClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

# Définir le split pour la validation croisée
SSplit = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

# Boucle sur les splits et les classifieurs pour obtenir la précision moyenne sur la validation
for train_index, test_index in SSplit.split(X_train, Y_train):
    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
    for clf in classifiers:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(clf.__class__.__name__, acc)


# ## Etape 5 : Evaluation

# L'objectif de cette étape va être d'évaluer notre modèle et de l'optimiser un maximum c'est à dire de choisir les meilleurs valeurs pour nos hyperparamètres (les hyperparamètres sont des paramètres que l'on choisit avant d'entraîner un modèle de machine learning, et qui affectent le processus d'apprentissage et la performance du modèle). Ces paramètres sont essentiels car leur optimisation permet d'obtenir les meilleurs paramètres pour un modèle d'apprentissage automatique et donc d'obtenir des prédictions plus précises et plus fiables.

# In[63]:


from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Étape de modélisation : Tester différents algorithmes
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({
    "Accuracy (Précision) Moyenne": cv_means,
    "CrossValerrors": cv_std,
    "Modèles": [
        "SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees",
        "GradientBoosting", "MultipleLayerPerceptron", "KNeighboors",
        "LogisticRegression", "LinearDiscriminantAnalysis"
    ]
})

g = sns.barplot(
    "Accuracy (Précision) Moyenne", "Modèles", data=cv_res, palette="Set3",
    orient="h", **{'xerr': cv_std}
)
g.set_xlabel("Accuracy (Précision) Moyenne")
g = g.set_title("Scores de validation croisée")


# In[64]:


import matplotlib.pyplot as plt
import seaborn as sns

# Définir les modèles et leurs noms
classifiers = [SVC(random_state=random_state),
               DecisionTreeClassifier(random_state=random_state),
               AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1),
               RandomForestClassifier(random_state=random_state),
               ExtraTreesClassifier(random_state=random_state),
               GradientBoostingClassifier(random_state=random_state),
               MLPClassifier(random_state=random_state),
               KNeighborsClassifier(),
               LogisticRegression(random_state=random_state),
               LinearDiscriminantAnalysis()]

model_names = ["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees",
               "GradientBoosting", "MultipleLayerPerceptron", "KNeighboors",
               "LogisticRegression", "LinearDiscriminantAnalysis"]

# Calculer la performance de chaque modèle
performance = []
for classifier in classifiers:
    cv_results = cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4)
    performance.append(cv_results.mean())

# Créer un graphique
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=performance, palette="Set3")
plt.xticks(rotation=90)
plt.xlabel("Modèles")
plt.ylabel("Accuracy")
plt.title("Performance de chaque modèle")

# Ajouter les légendes correspondant à la valeur de chaque modèle
for i, v in enumerate(performance):
    plt.text(i, v, "{:.2f}".format(v), color='black', ha='center', fontweight='bold')

plt.show()


# L'avantage de cette deuxième option réside dans sa méthode de calcul. En effet, ici pour chaque split une précision sera calculée et la précision du modèle correspondra à la moyenne des précisions du modèle sur chacune des données splitées. Cela permet de nous rassurer sur la performance du modèle, de limiter l'overfitting et d'annoncer que la random forest semble être le modèle le plus précis avec 84% de précision. Nous pouvons toutefois l'améliorer en modifiant ses hyperparamètres. 

# Je me suis permis d'ajouter ce tableau qui contient l'importance de chaque feature dans le modèle de forêt aléatoire (random_forest) pour voir le poids de chaque feature dans le modèle.

# In[65]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# entraîner le modèle sur les données d'entraînement
random_forest.fit(X_train, Y_train)
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(random_forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending= False).set_index('feature')

importances.head(15)


# In[66]:


from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à tester pour la Random Forest
rf_param_grid = {
    'n_estimators': [100, 300, 500, 800, 1200],
    'max_depth': [5, 8, 15, 25, 30],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Instancier la Random Forest
rf = RandomForestClassifier()

# Recherche par grille pour optimiser les hyperparamètres de la Random Forest
rf_grid_search = GridSearchCV(estimator = rf, param_grid = rf_param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Entraîner la Random Forest avec les hyperparamètres optimisés
rf_grid_search.fit(x_train, y_train)

# Afficher les hyperparamètres optimisés
print("Meilleurs hyperparamètres pour la Random Forest: ", rf_grid_search.best_params_)

# Afficher la précision de la Random Forest avec les hyperparamètres optimisés
y_pred = rf_grid_search.predict(x_val)
acc = accuracy_score(y_val, y_pred)
print("Précision de la Random Forest avec les hyperparamètres optimisés: ", acc)


# Le modèle le plus performant que nous pouvons obtenir se base sur la Random Forest est à une précision de 91,75%. Les meilleurs hyperparamètres sont : une max_depth de 30, un min_samples_leaf de 2, un min_samples_split de 2 et un n_estimators de 100. Voilà le modèle le plus performant que j'ai pu obtenir.

# In[67]:


from sklearn.metrics import f1_score

# Prédire les valeurs cibles pour le jeu de validation
y_pred = rf_grid_search.predict(x_val)

# Calculer le score F1 avec les hyperparamètres optimisés pour la Random Forest
f1 = f1_score(y_val, y_pred, average='weighted')

print("Score F1 de la Random Forest avec les hyperparamètres optimisés: ", f1)


# In[75]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Utiliser le modèle optimisé pour prédire les classes sur le jeu de validation
y_pred = rf_grid_search.predict(x_val)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_val, y_pred)

# Afficher la matrice de confusion sous forme de matrice
print("Matrice de confusion : ")
print(np.array2string(conf_matrix, separator=', '))

# Afficher la matrice de confusion sous forme graphique
plot_confusion_matrix(rf_grid_search, x_val, y_val, cmap=plt.cm.Blues)
plt.show()


# Dans cet exemple, la matrice de confusion montre que le modèle a prédit 110 cas de la classe 0 (pas de survie) correctement et 62 cas de la classe 1 (survie) correctement. Cependant, il a également fait 14 erreurs en prédisant une survie alors qu'il n'y en avait pas et 8 erreurs en prédisant l'absence de survie alors qu'il y en avait une.

# In[76]:


from sklearn.metrics import plot_roc_curve

# Obtenir le meilleur modèle après la recherche par grille
best_rf = rf_grid_search.best_estimator_

# Afficher la courbe ROC pour le meilleur modèle
plot_roc_curve(best_rf, x_val, y_val)
plt.show()


# In[77]:


# Obtenir le meilleur modèle après la recherche par grille
best_rf = rf_grid_search.best_estimator_

# Afficher la courbe ROC pour le meilleur modèle
plot_roc_curve(best_rf, x_val, y_val)
plt.show()

# Obtenir les valeurs des faux positifs, des vrais positifs et des seuils
fpr, tpr, thresholds = roc_curve(y_val, best_rf.predict_proba(x_val)[:,1])

# Afficher les valeurs des faux positifs et des vrais positifs
print("Faux positifs : ", fpr)
print("Vrais positifs : ", tpr)


# Un AUC de 0,95 est un excellent score et indique que le modèle est très performant pour la classification binaire. Pour finir, il est donc possible de conclure que le modèle de Random Forest étudié a tendance à bien prédire les passagers décédés ainsi que la prédiction des passagers ayant survécu.

# In[ ]:




