import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from scipy.stats import wilcoxon, ttest_1samp, ttest_ind
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.contingency_tables import mcnemar

def test_lambda_la_puterea_2(df, var_categorica1, var_categorica2):
    # Creează o tabelă de contingență 2x2
    tabela_contingenta = pd.crosstab(df[var_categorica1], df[var_categorica2])

    # Aplică testul lambda la puterea 2
    # Calculul testului Lambda la puterea 2
    n = tabela_contingenta.values.sum()
    C = len(tabela_contingenta.index)
    R = len(tabela_contingenta.columns)

    # Testul Lambda la puterea 2
    lambda_value = (n * (tabela_contingenta.values.diagonal().sum()) - (1 / C) * (
        tabela_contingenta.values.sum()) ** 2) / \
                   (n * (1 - (1 / C)) * (R - 1))

    # Afișează rezultatele
    print(f"Testul λ la puterea 2 pentru {var_categorica1} și {var_categorica2}: {lambda_value}")
    print("Tabelul de contingență 2x2:")
    print(tabela_contingenta)


def test_fisher_exact(df, var_categorica1, var_categorica2):
    # Alege cele două variabile categorice pentru analiză
    var1_values = df[var_categorica1].values
    var2_values = df[var_categorica2].values

    # Creează o tabelă de contingență 2x2
    tabela_contingenta = pd.crosstab(var1_values, var2_values)

    # Verifică dacă dimensiunea tabelului este potrivită
    if tabela_contingenta.shape == (2, 2):
        # Calculează testul Fischer Exact
        odds_ratio, p_value = fisher_exact(tabela_contingenta)

        # Afișează rezultatele
        print(f"Odds Ratio pentru {var_categorica1} și {var_categorica2}: {odds_ratio}")
        print(f"P-value pentru {var_categorica1} și {var_categorica2}: {p_value}")
    else:
        print(
            f"Tabela de contingență pentru variabilele {var_categorica1} și {var_categorica2} nu are dimensiunea (2, 2). Nu s-a putut efectua Testul Exact Fischer")



def test_mann_whitney(df, var_continua1, var_continua2):
    # Selectează valorile pentru variabilele continue
    var1_values = df[var_continua1].values
    var2_values = df[var_continua2].values

    # Calculează testul Mann-Whitney
    stat, p_value = mannwhitneyu(var1_values, var2_values, alternative='two-sided')

    # Afișează rezultatele
    print(f"Testul Mann-Whitney pentru {var_continua1} și {var_continua2}:")
    print(f"Statistică de test: {stat}")
    print(f"p-value: {p_value}")
    print()


def test_wilcoxon(df, var_continua1, var_continua2):
    # Implementare test Wilcoxon pentru două eșantioane dependente
    statistic, p_value = wilcoxon(df[var_continua1], df[var_continua2])
    print(f"Testul Wilcoxon pentru {var_continua1} și {var_continua2}:")
    print(f"Statistic: {statistic}")
    print(f"P-Value: {p_value}")
    print("\n")

def test_sign(df, var_continua1, var_continua2):
    # Implementare test semnului pentru diferența dintre două eșantioane dependente
    result = sign_test(df[var_continua1] - df[var_continua2])
    statistic, p_value = result if len(result) == 2 else (*result, None)
    print(f"Testul semnului pentru diferența dintre {var_continua1} și {var_continua2}:")
    print(f"Statistic: {statistic}")
    print(f"P-Value: {p_value}")
    print("\n")

def test_mcnemar(df, var_categorica1, var_categorica2):
    # Implementare test McNemar pentru două eșantioane dependente (variabile categorice)
    tabela_contingenta = pd.crosstab(df[var_categorica1], df[var_categorica2])
    result = mcnemar(tabela_contingenta)
    statistic, p_value = result.statistic, result.pvalue
    print(f"Testul McNemar pentru {var_categorica1} și {var_categorica2}:")
    print(f"Statistic: {statistic}")
    print(f"P-Value: {p_value}")
    print("\n")


def test_comparare_medie(df, var_continua, valoare_comparata):
    """
    Testează dacă media unei variabile continue este semnificativ diferită de o valoare dată.

    :param df: DataFrame-ul cu date
    :param var_continua: Numele variabilei continue de testat
    :param valoare_comparata: Valoarea de referință pentru comparație
    """
    # Filtrăm doar valorile non-nule ale variabilei continue
    subset = df[var_continua].dropna()

    # Realizăm testul t pentru două eșantioane independente
    t_statistic, p_value = ttest_1samp(subset, valoare_comparata)

    # Afișăm rezultatele testului
    print(f"\nTestul t pentru comparația mediei variabilei {var_continua} cu valoarea {valoare_comparata}:")
    print(f"Media observată: {subset.mean()}")
    print(f"Valoarea de comparație: {valoare_comparata}")
    print(f"Test statistic: {t_statistic}")
    print(f"p-value: {p_value}")

def test_t_doua_eshantioane(df, var_continua1, var_continua2):
    """
    Testează dacă două eșantioane independente au medii semnificativ diferite.

    :param df: DataFrame-ul cu date
    :param var_continua1: Numele primei variabile continue
    :param var_continua2: Numele celei de-a doua variabile continue
    """
    # Filtrăm valorile non-nule ale celor două variabile continue
    subset1 = df[var_continua1].dropna()
    subset2 = df[var_continua2].dropna()

    # Realizăm testul t pentru două eșantioane independente
    t_statistic, p_value = ttest_ind(subset1, subset2)

    # Afișăm rezultatele testului
    print(f"\nTestul t pentru două eșantioane independente între variabilele {var_continua1} și {var_continua2}:")
    print(f"Media variabilei {var_continua1}: {subset1.mean()}")
    print(f"Media variabilei {var_continua2}: {subset2.mean()}")
    print(f"Test statistic: {t_statistic}")
    print(f"p-value: {p_value}")


def test_t_doua_eshantioane_independente(df, var_continua, var_categorica, grup1, grup2):

    grup1_values = df[df[var_categorica] == grup1][var_continua]
    grup2_values = df[df[var_categorica] == grup2][var_continua]

    statistic, p_value = ttest_ind(grup1_values, grup2_values, equal_var=True)
    return statistic, p_value