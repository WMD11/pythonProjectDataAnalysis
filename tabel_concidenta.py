import pandas as pd
from scipy.stats import chi2_contingency

def analiza_coeficient_correlatie_phi(df, var_categorica1, var_categorica2):
    # Creează o tabelă de contingență pentru cele două variabile
    tabela_contingenta = pd.crosstab(df[var_categorica1], df[var_categorica2])

    # Calculul coeficientului de corelație phi
    chi2, p, dof, expected = chi2_contingency(tabela_contingenta)

    # Calculul coeficientului de corelație phi
    phi_coefficient = (chi2 / len(df))**0.5

    # Afișează rezultatele
    print(f"Coeficientul de corelație phi între {var_categorica1} și {var_categorica2} este: {phi_coefficient}")
