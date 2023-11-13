import pandas as pd
from coeficienti_corelatii import coef_pearson, coef_spearman, coef_kendall
from tabel_concidenta import analiza_coeficient_correlatie_phi
from indici_asociere import analiza_indici_asociere
from teste_statistice import test_lambda_la_puterea_2, test_fisher_exact, test_mann_whitney, test_wilcoxon, test_sign, test_mcnemar, test_comparare_medie, test_t_doua_eshantioane, test_t_doua_eshantioane_independente

# Încarcă setul de date
df = pd.read_csv('dataset.csv')

# Cerința 1: Coeficientul de corelație Pearson
coef_pearson(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerința 2: Coeficientul de corelație Spearman
coef_spearman(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerința 3: Coeficientul de corelație Kendall
coef_kendall(df, 'Duration', 'Cost of Travel(Entire Trip)')


# Cerința 4: Indici de asociere
# Alegem două variabile categorice pentru analiză
var_categorica1 = 'Sex'
var_categorica2 = 'Stay'

# Apelul funcției
analiza_indici_asociere(df, var_categorica1, var_categorica2)

# Cerința 5: Tabel de contingență și test chi-square

# Alegem două variabile categorice pentru analiză
var_categorica1 = 'Sex'
var_categorica2 = 'Stay'

# Apelul funcției
analiza_coeficient_correlatie_phi(df, var_categorica1, var_categorica2)

# Cerința 6: Testul λ la puterea 2 și Tabele 2x2
var_categorica1 = 'Sex'
var_categorica2 = 'Stay'

# Apelul funcției
test_lambda_la_puterea_2(df, var_categorica1, var_categorica2)

# Cerința 7: Testul Exact Fischer
test_fisher_exact(df, 'Age', 'Duration')

# Cerința 8: Testul Mann-Whitney
test_mann_whitney(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerința 9: Testul Wilcoxon
test_wilcoxon(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerința 10: Testul semnului
test_sign(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerința 11: Testul McNemar
test_mcnemar(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerinta 12: Test pentru compararea mediei unui eșantion, cu o valoare dată
test_comparare_medie(df, 'Duration', df['Duration'].median())

# Cerinta 13: Testul T pentru două eșantioane independente
test_t_doua_eshantioane(df, 'Duration', 'Cost of Travel(Entire Trip)')

# Cerinta 14:  Testul T pentru două eșantioane independente
var_continua = 'Duration'
var_categorica = 'Mode of Travel'
grup1 = 'Flight'
grup2 = 'Car'

# Apelul funcției
statistic, p_value = test_t_doua_eshantioane_independente(df, var_continua, var_categorica, grup1, grup2)

# Afișarea rezultatelor
print(f"Testul T pentru două eșantioane independente între {grup1} și {grup2}:")
print(f"Statistic: {statistic}")
print(f"p-value: {p_value}")