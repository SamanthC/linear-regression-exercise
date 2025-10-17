#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'exploration des données - Prédiction de Salaires
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Chargement des données
print("=" * 80)
print("EXPLORATION DES DONNÉES - PRÉDICTION DE SALAIRES")
print("=" * 80)

df = pd.read_csv('data/employee_salaries.csv')

# 1. INFORMATIONS GÉNÉRALES
print("\n" + "=" * 80)
print("1. INFORMATIONS GÉNÉRALES")
print("=" * 80)

print(f"\nNombre de lignes : {df.shape[0]:,}")
print(f"Nombre de colonnes : {df.shape[1]}")

print("\n--- Structure du DataFrame ---")
print(df.info())

print("\n--- Aperçu des premières lignes ---")
print(df.head(10))

print("\n--- Aperçu des dernières lignes ---")
print(df.tail(5))

# 2. STATISTIQUES DESCRIPTIVES
print("\n" + "=" * 80)
print("2. STATISTIQUES DESCRIPTIVES")
print("=" * 80)

print("\n--- Variables Numériques ---")
print(df.describe())

print("\n--- Variables Catégorielles ---")
colonnes_cat = df.select_dtypes(include=['object']).columns
for col in colonnes_cat:
    print(f"\n{col.upper()} - {df[col].nunique()} valeurs uniques")
    print(df[col].value_counts().head(10))

# 3. VALEURS MANQUANTES
print("\n" + "=" * 80)
print("3. ANALYSE DES VALEURS MANQUANTES")
print("=" * 80)

valeurs_manquantes = df.isnull().sum()
pourcentage_manquant = (valeurs_manquantes / len(df)) * 100
missing_df = pd.DataFrame({
    'Colonne': valeurs_manquantes.index,
    'Nb_Manquants': valeurs_manquantes.values,
    'Pourcentage': pourcentage_manquant.values
})
missing_df = missing_df[missing_df['Nb_Manquants'] > 0].sort_values('Nb_Manquants', ascending=False)

if len(missing_df) > 0:
    print("\n⚠️  Colonnes avec valeurs manquantes :")
    print(missing_df.to_string(index=False))
else:
    print("\n✅ Aucune valeur manquante dans le dataset !")

# 4. DOUBLONS
print("\n" + "=" * 80)
print("4. ANALYSE DES DOUBLONS")
print("=" * 80)

nb_doublons = df.duplicated().sum()
print(f"\nNombre de lignes dupliquées : {nb_doublons}")
if nb_doublons > 0:
    print("⚠️  Des doublons ont été détectés !")
else:
    print("✅ Aucun doublon détecté !")

# 5. DISTRIBUTION DE LA VARIABLE CIBLE
print("\n" + "=" * 80)
print("5. ANALYSE DE LA VARIABLE CIBLE : SALAIRE_ANNUEL")
print("=" * 80)

print("\n--- Statistiques du salaire ---")
print(f"Moyenne : {df['salaire_annuel'].mean():,.2f} €")
print(f"Médiane : {df['salaire_annuel'].median():,.2f} €")
print(f"Écart-type : {df['salaire_annuel'].std():,.2f} €")
print(f"Min : {df['salaire_annuel'].min():,.2f} €")
print(f"Max : {df['salaire_annuel'].max():,.2f} €")
print(f"Q1 (25%) : {df['salaire_annuel'].quantile(0.25):,.2f} €")
print(f"Q3 (75%) : {df['salaire_annuel'].quantile(0.75):,.2f} €")

# Détection des valeurs aberrantes (outliers) avec la méthode IQR
Q1 = df['salaire_annuel'].quantile(0.25)
Q3 = df['salaire_annuel'].quantile(0.75)
IQR = Q3 - Q1
outliers_inf = (df['salaire_annuel'] < (Q1 - 1.5 * IQR)).sum()
outliers_sup = (df['salaire_annuel'] > (Q3 + 1.5 * IQR)).sum()
print(f"\n--- Valeurs aberrantes (méthode IQR) ---")
print(f"Outliers inférieurs : {outliers_inf}")
print(f"Outliers supérieurs : {outliers_sup}")
print(f"Total outliers : {outliers_inf + outliers_sup} ({((outliers_inf + outliers_sup)/len(df)*100):.2f}%)")

# 6. CORRÉLATIONS ENTRE VARIABLES NUMÉRIQUES
print("\n" + "=" * 80)
print("6. MATRICE DE CORRÉLATION")
print("=" * 80)

colonnes_num = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[colonnes_num].corr()

print("\n--- Corrélations avec salaire_annuel (en ordre décroissant) ---")
corr_avec_salaire = correlation_matrix['salaire_annuel'].sort_values(ascending=False)
print(corr_avec_salaire)

print("\n--- Top 5 des corrélations les plus fortes avec le salaire ---")
top_corr = corr_avec_salaire[1:6]  # Exclure la corrélation avec elle-même
for col, corr in top_corr.items():
    print(f"{col:30s} : {corr:+.4f}")

# 7. ANALYSE PAR CATÉGORIES
print("\n" + "=" * 80)
print("7. SALAIRE MOYEN PAR CATÉGORIE")
print("=" * 80)

for col in colonnes_cat:
    print(f"\n--- Salaire moyen par {col.upper()} ---")
    salaire_par_cat = df.groupby(col)['salaire_annuel'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    salaire_par_cat.columns = ['Moyenne', 'Médiane', 'Nombre']
    salaire_par_cat['Moyenne'] = salaire_par_cat['Moyenne'].apply(lambda x: f"{x:,.2f} €")
    salaire_par_cat['Médiane'] = salaire_par_cat['Médiane'].apply(lambda x: f"{x:,.2f} €")
    print(salaire_par_cat.to_string())

# 8. RÉSUMÉ FINAL
print("\n" + "=" * 80)
print("8. RÉSUMÉ DE L'EXPLORATION")
print("=" * 80)

print(f"""
Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes

Variables numériques : {len(colonnes_num)}
  - {', '.join(colonnes_num)}

Variables catégorielles : {len(colonnes_cat)}
  - {', '.join(colonnes_cat)}

Variable cible : salaire_annuel
  - Plage : {df['salaire_annuel'].min():,.0f} € → {df['salaire_annuel'].max():,.0f} €
  - Moyenne : {df['salaire_annuel'].mean():,.2f} €

Qualité des données :
  - Valeurs manquantes : {df.isnull().sum().sum()} ({(df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100):.2f}%)
  - Doublons : {nb_doublons}
  - Outliers (salaire) : {outliers_inf + outliers_sup} ({((outliers_inf + outliers_sup)/len(df)*100):.2f}%)

Corrélations les plus fortes avec le salaire :
""")

for i, (col, corr) in enumerate(corr_avec_salaire[1:4].items(), 1):
    print(f"  {i}. {col:30s} : {corr:+.4f}")

print("\n" + "=" * 80)
print("Exploration terminée ! ✅")
print("=" * 80)

