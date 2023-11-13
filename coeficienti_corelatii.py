import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency

def coef_pearson(df, var1, var2):
    pearson_corr, _ = pearsonr(df[var1], df[var2])
    print(f"Coeficientul de corelație Pearson între '{var1}' și '{var2}' este: {pearson_corr}")

def coef_spearman(df, var1, var2):
    spearman_corr, _ = spearmanr(df[var1], df[var2])
    print(f"Coeficientul de corelație Spearman între '{var1}' și '{var2}' este: {spearman_corr}")

def coef_kendall(df, var1, var2):
    kendall_corr, _ = kendalltau(df[var1], df[var2])
    print(f"Coeficientul de corelație Kendall între '{var1}' și '{var2}' este: {kendall_corr}")
