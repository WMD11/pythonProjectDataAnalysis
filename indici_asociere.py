import pandas as pd
import numpy as np
from scipy import stats


def analiza_indici_asociere(df, var_categorica1, var_categorica2):
    tabela_contingenta = pd.crosstab(df[var_categorica1], df[var_categorica2])
    chi2, _, _, _ = stats.chi2_contingency(tabela_contingenta)

    numar_linii = tabela_contingenta.shape[0]
    numar_coloane = tabela_contingenta.shape[1]
    min_dimensiune = min(numar_linii, numar_coloane)

    v_cramer = np.sqrt(chi2 / (len(df) * (min_dimensiune - 1)))

    print(f"Indicele de asociere Cramer's V între {var_categorica1} și {var_categorica2} este: {v_cramer}")
