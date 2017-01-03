import numpy as np
import pandas as pd
import csv as csv
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as p
from scipy.stats import ttest_ind
from numpy import genfromtxt


plt.style.use('ggplot')

#abre o csv e insere no pandas
filename = 'D:/Talita/Pessoal/Udacity/Analise de Dados/Projeto Final/titanic_data.csv'
titanic = pd.read_csv(filename)

#verifica a os dados
#print titanic

def gera_grafico_barras (variavel):

    variavel["Survived"].plot(kind='bar', title="Percentual de Sobreviventes")
    plt.axhline(0, color='k')
    plt.show()

    pass

def gera_grafico_2d(x, col, data):

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(x=col, y="Survived", hue=x, data=data, split=True,
                   inner="quart")
    sns.despine(left=True)
    plt.show()

    pass

def calcula_numero_sobreviventes(coluna):

    agrupamento_quantidade = titanic.groupby([coluna]).count()
    agrupamento_sobreviventes = titanic.groupby([coluna]).sum()
    agrupamento_sobreviventes_taxa = agrupamento_sobreviventes / agrupamento_quantidade

    return agrupamento_sobreviventes_taxa

def calcula_numero_sobreviventes_2d(coluna1, coluna2):

    agrupamento_quantidade = titanic.groupby([coluna1,coluna2]).count()
    agrupamento_sobreviventes = titanic.groupby([coluna1,coluna2]).sum()
    agrupamento_sobreviventes_taxa = agrupamento_sobreviventes / agrupamento_quantidade

    return agrupamento_sobreviventes_taxa

#verifica quantiadde de nulos e datatype dos dados
#print titanic.info()
#as variaveis numericas como passarid, survived foram importadas como int. Existem nulos na variavel idade e tipo de cabine

#retira nulo da variavel idade inserindo a media por classe e sexo



sexo = ['female','male']

mediana_idade = np.zeros((2,3))


#print titanic[titanic['Survived'] == 1][['Age']]

for i in sexo:

    if i == 'female':
        cod_sexo = 0
    else:
        cod_sexo = 1
    for j in range(0, 3):
        mediana_idade[cod_sexo,j] = titanic[(titanic['Sex'] == i) & (titanic['Pclass'] == j+1)]['Age'].dropna().median()


titanic['Idade_preenchimento'] = titanic['Age']

for i in sexo:
    if i == 'female':
        cod_sexo = 0
    else:
        cod_sexo = 1

    for j in range(0, 3):
        titanic.loc[(titanic.Age.isnull()) & (titanic.Sex == i) & (titanic.Pclass == j+1),'Idade_preenchimento'] = mediana_idade[cod_sexo,j]

titanic['Age'] = titanic['Idade_preenchimento']

#print titanic[titanic['Age'].isnull()][['Sex', 'Pclass', 'Age', 'Idade_preenchimento']].head(10)

print titanic
#verifica estatistica das colunas
print titanic.describe()
#Percebe-se que existiam 891 passageiros a bordo.  A idade das pessoas variavam de bebes a idosos com 80 anos. A classe vai de 0 a 3.
#O preco da tarifa variou de 0 a 512,32. Sera que e correto existirem tarifas zeradas?
# O numer de parentes a bordo varia de 0 a 6. O numero de irmaos/esposos varia de 0 a 8.
#Qual o numero de sobreviventes?

#calculando numero de sobreviventes
quantidade_sobreviventes = titanic[["Survived"]].sum()
quantidade_passageiros = titanic[["Survived"]].count()
psobreviventes = quantidade_sobreviventes /quantidade_passageiros

#print 'Quantidade de Sobreviventes = ', quantidade_sobreviventes
#print '% sobreviventes = ' , psobreviventes

#verificar tarifas zeradas
#print titanic.loc[lambda titanic1 : titanic.Fare == 0, :]

#percebe-se que todos os passageiros com preco da passagem igual a zero sao homens. Nao consegui perceber nenhum erro.
# Percebi que apenas um passageiro sobreviveu, qual sera a taxa de sobrevivencia para pessoas com passagem igual a zero?
quantidade_sobreviventes_t0 = titanic.loc[lambda titanic1 : titanic.Fare == 0, :].sum()
quantidade_passageiros_t0 = titanic.loc[lambda titanic1 : titanic.Fare == 0, :].count()
psobreviventes_t0 =  quantidade_sobreviventes_t0["Survived"]/quantidade_passageiros_t0["Survived"]
#print 'Quantidade de Sobreviventes = ', quantidade_sobreviventes_t0["Survived"]
#print '% sobreviventes = ' , psobreviventes_t0

#taxa de sobrevivencia e extremamente baixa para pessoas com preco da passagem igual a zero, ou seja, se voce nao pagou passagem e provavel que voce nao sobreviva


#Qual a taxa de sobrevivente por sexo?

agrupamento_sexo_sobreviventes_taxa = calcula_numero_sobreviventes("Sex")
#print agrupamento_sexo_sobreviventes_taxa["Survived"]
#gera_grafico_barras (agrupamento_sexo_sobreviventes_taxa)

#percebe-se que a taxa de sobrevivencia ' muito mais alta entre as mulheres do que homens, ou seja, se voce for mulher e mais provavel que voce sobreviva.

#Qual a taxa de sobrevivencia por idade?

agrupamento_idade_sobreviventes_taxa = calcula_numero_sobreviventes("Age")
#print agrupamento_idade_sobreviventes_taxa["Survived"]
#gera_grafico_barras (agrupamento_idade_sobreviventes_taxa)
#percebe-se que criancas de ate um ano e idosos acima de 63 anos tem 100 de chance de sobrevivencia. mas nao ficou muito claro para outras idades.
#
# Vou criar faixas de idade para averiguar se existe alguma relacao

bins = [0,10,20,30,40,50,60,80]
group_names = ['0 a 9','10 a 19','20 a 29','30 a 39','40 a 49','50 a 59', '>60']
faixa_idade = pd.cut(titanic['Age'], bins, labels=group_names)
titanic['faixa_idade'] = pd.cut(titanic['Age'], bins, labels=group_names)


agrupamento_idade_sobreviventes_taxa = calcula_numero_sobreviventes("faixa_idade")
#print agrupamento_idade_sobreviventes_taxa["Survived"]
#gera_grafico_barras(agrupamento_idade_sobreviventes_taxa)

#vou tentar um histograma de idade para verificar como se comporta:
titanic['Age'].hist(bins=16, range=(0,80), alpha = .5)
p.show()
#Com as faixas de idade continuamos com o mesmo resultado, vou cruzar idade com sexo para verificar se por sexo existe alguma relacao

agrupamento_idade_sexo_sobreviventes_taxa = calcula_numero_sobreviventes_2d("faixa_idade","Sex")
print agrupamento_idade_sexo_sobreviventes_taxa["Survived"]
gera_grafico_2d("Sex", "faixa_idade", titanic)





#Se voce e homem e mais provavel que sobreviva se tiver menos de 10 anos.


#Vou verificar se existe alguma relacao entre a classe social

agrupamento_classe_sobreviventes_taxa = calcula_numero_sobreviventes("Pclass")
#print agrupamento_classe_sobreviventes_taxa["Survived"]
#gera_grafico_barras(agrupamento_classe_sobreviventes_taxa)



#Vou cruzar sexo com classe social para verificar se por sexo existe alguma relacao

agrupamento_classe_sexo_sobreviventes_taxa = calcula_numero_sobreviventes_2d("Pclass","Sex")
#print agrupamento_classe_sexo_sobreviventes_taxa["Survived"]
gera_grafico_2d("Sex", "Pclass", titanic)


#96 porcento das mulheres da primeira classe e 92 da segunda classe sobreviveram. No entanto apenas 50 das mulheres da 3 classe sobrevivera.. Portanto, se voce e mulher suas chances de sobreviver foram sao mais altas mas se for da 3 classe isso cai drasticamente.
#Para os homens tambem existe uma relacao, mas mesmo homens da primeira classe tem baixas chances de sobrevivencia. E as chances de sobreviver e quase a mesma entre homens de segunda e terceira classe

#60 dos passageiros da primeira classe sobreviveram, quanto mais alta a classe maior a chance de sobrevivencia.


#Vou verificar se existe alguma relacao entre sobrevivencia e porto de embarque

agrupamento_porto_sobreviventes_taxa = calcula_numero_sobreviventes("Embarked")
#print agrupamento_porto_sobreviventes_taxa["Survived"]
#gera_grafico_barras(agrupamento_porto_sobreviventes_taxa)

#aproximadamente 55 dos passageiros que embaracaram no porto  de Cherbourg sobreviveram e contra 33 dos que embarcaram em Southampton. Portanto se o passageiro embarcou em Cherbourg ele possui mais chances de sobreviver.


#Referencia
#http://stackoverflow.com/questions/22649693/drop-rows-with-all-zeros-in-pandas-data-frame
#http://pandas.pydata.org/pandas-docs/stable/indexing.html
#http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html
#http://chrisalbon.com/python/pandas_binning_data.html
#http://stackoverflow.com/questions/37955881/multiple-bar-plots-from-pandas-dataframe
#https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii
#http: // seaborn.pydata.org / examples / grouped_violinplots.html


tt1 = titanic[titanic['Sex'] == 'female']
tt2 = titanic[titanic['Sex'] == 'male']

print ttest_ind(tt1['Survived'], tt2['Survived'])