---
title: "Modelagem de Crédito"
subtitle: "Engenharia de variáveis para previsão de inadimplência"
excerpt: "Grid is the very first CSS module created specifically to solve the layout problems we’ve all been hacking our way around for as long as we’ve been making websites."
date: 2023-06-10
author: "Mauricio Pozzebon"
draft: false
tags:
- hugo-site
categories:
- Python
- Machine Learning
# layout options: single or single-sidebar
layout: single
links:
- icon: github
  icon_pack: fab
  name: code
  url: https://github.com/allisonhorst/palmerpenguins/

---

<!--{{< here >}}-->

### Tão importante quanto a disponibilidade de crédito é a capacidade de prever um possível *default* e assim ter as provisões necessárias para tal.



---

### Quais as pistas da inadimplência?

Prever **inadimplência** é um problema clássico nas instituições de crédito, ainda mais quando não se tem informações históricas a respeito do cliente. Como saber se um cliente novo não dará "calote"?   

Este projeto desenvolve um modelo de previsão de inadimplência a partir da exploração e engenharia de variáveis de duas *databases* de empréstimos anonimizados. O modelo então é utilizado para prever o *default* em uma lista de clientes **futuros**.

Meu *machine learning pipeline*: importar datasets &#x2192; inspeção &#x2192; preprocessamento &#x2192; análise exploratória &#x2192; modelagem &#x2192; avaliação &#x2192; previsão.

<!--{{< figure src="css-grid-cover.png" alt="Traditional right sidebar layout" caption="A visual example of the traditional right sidebar layout" >}}-->

---

### Código

Primeiramente vamos **importar** os módulos já sabendo que se trata de um problema de classificação:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
```

Vamos trabalhar em dois *datasets*, um com informações de cadastro (`base cadastral`) e outro (`base_pagamentos_desenvolvimento`) que utilizaremos para treinar o modelo:

```python
base_cadastral = pd.read_csv("base_cadastral.csv")
base_pagamentos_desenvolvimento = pd.read_csv("base_pagamentos_desenvolvimento.csv")
```
Iniciando a inspeção:

```python
print(base_cadastral)
print(base_pagamentos_desenvolvimento)
```
Uma rápida olhada em valores nulos para termos ideia do quê trabalhar na base:

```python
print(base_pagamentos_desenvolvimento.isna().sum())
print(base_cadastral.isna().sum())
```

Conclusão, todos os valores nulos pertencem à base cadastral.

A documentação da base indica que `FLAG_PF = X` se pesoa física e `FLAG_PF = NaN` se pessoa jurídica. Portanto:

```python
def flag(x):
    if x == 'X':
        return 'PF'
    else:
        return 'PJ'

base_cadastral['FLAG'] = base_cadastral['FLAG_PF'].apply(lambda x: flag(x))
base_cadastral.drop(columns='FLAG_PF', inplace = True)
```
Agora é necessário tratar `DDD`,`SEGMENTO_INDUSTRIAL`, `PORTE` e `CEP_2_DIG`.

Começando pelo `PORTE`, as categorias são vistas em:

```python
base_cadastral['PORTE'].value_counts()
```
Vamos criar uma nova categoria `indefinido`:

```python
def substituir_nulos(df):
    base_cadastral['PORTE'].fillna('indefinido', inplace=True)
    return base_cadastral

base_cadastral = substituir_nulos(base_cadastral)
```
Apliquei a mesma lógica para `SEGMENTO_INDUSTRIAL`:

```python
def substituir_nulos(df):
    base_cadastral['SEGMENTO_INDUSTRIAL'].fillna('seg_indefinido', inplace=True)
    return base_cadastral

base_cadastral = substituir_nulos(base_cadastral)
```
Da mesma forma, tratamos a coluna `DDD`:

```python
def substituir_nulos(df):
    base_cadastral['DDD'].fillna('ddd_indefinido', inplace=True)
    return base_cadastral

base_cadastral = substituir_nulos(base_cadastral)
```

Por fim, a coluna `CEP_2_DIG`:

```python
def substituir_nulos(df):
    base_cadastral['CEP_2_DIG'].fillna('cep_indefinido', inplace=True)
    return base_cadastral

base_cadastral = substituir_nulos(base_cadastral)
```
Agora vamos transformaros valores de `DDD`e `CEP_2_DIG` em categorias (regiões e Estado respectivamente):

```python
def regiao(ddd):
    if ddd in ['61']:
        return 'Centro-Oeste'
    elif ddd in ['62', '64', '65', '66', '67']:
        return 'Centro-Oeste'
    elif ddd in ['82']:
        return 'Nordeste'
    elif ddd in ['71', '73', '74', '75', '77', '85', '88', '98', '99', '83', '81', '87', '86', '89', '84', '79']:
        return 'Nordeste'
    elif ddd in ['68', '96', '92', '97', '91', '93', '94', '69', '95', '63']:
        return 'Norte'
    elif ddd in ['27', '28', '31', '32', '33', '34', '35', '37', '38', '21', '22', '24', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
        return 'Sudeste'
    elif ddd in ['41', '42', '43', '44', '45', '46', '51', '53', '54', '55', '47', '48', '49']:
        return 'Sul'
    else:
        return 'DDD inválido'

base_cadastral['REGIAO'] = base_cadastral['DDD'].apply(regiao)
```
```python
def cep(x):
    if x.startswith('0'):
        return 'São Paulo'
    elif x.startswith('1'):
        return 'Interior de São Paulo'
    elif x.startswith('2'):
        return 'Rio de Janeiro ou Espírito Santo'
    elif x.startswith('3'):
        return 'Minas Gerais'
    elif x.startswith('4'):
        return 'Bahia ou Sergipe'
    elif x.startswith('5'):
        return 'Pernambuco, Alagoas, Rio Grande do Norte ou Paraíba'
    elif x.startswith('6'):
        return 'Maranhão, Acre, Pará, Amapá, Roraima, Ceará ou Amazonas'
    elif x.startswith('7'):
        return 'Mato Grosso do Sul, Tocantins, Mato Grosso, Goiás, Rondônia ou Distrito Federal'
    elif x.startswith('8'):
        return 'Paraná ou Santa Catarina'
    elif x.startswith('9'):
        return 'Rio Grande do Sul'
    else:
        return 'Região não identificada'

base_cadastral['CEP'] = base_cadastral['CEP_2_DIG'].apply(cep)
```

Vimos que a base de treino é pobre em termos de informações individuais. Sendo assim, vamos fundi-la com as informações cadastrais e "enriquecer" o *dataset* de treino para cada cliente (`ID_CLIENTE`). Em seguida, excluir os dados de cadastro sem correspondência:

```python
base_treino = pd.merge(base_pagamentos_desenvolvimento,base_cadastral,on=['ID_CLIENTE'], how='outer')
base_treino.drop(base_treino.index[77414:], inplace=True)
```
Uma olhada nos dados faltantes agora na base de treino completa:

```python
print(base_treino.isna().sum())
```
Temos nulos apenas no domínio de e-mail, uma informação irrelevante para o problema. Agora Podemos excluir as colunas que com certeza **não** usaremos no modelo(`ID_CLIENTE` e `DOMINIO_EMAIL`):

```python
base_treino.drop(columns=['ID_CLIENTE', 'DOMINIO_EMAIL'], inplace = True)
```

Vamos identificar os inadimplentes a partir do critério dado na documentação (pagamento atrasado em mais de 5 dias), atribuindo o valor 1 ou 0:

```python
base_treino['DATA_PAGAMENTO'] = pd.to_datetime(base_treino['DATA_PAGAMENTO'])
base_treino['DATA_VENCIMENTO'] = pd.to_datetime(base_treino['DATA_VENCIMENTO'])

base_treino['INADIMPLENTE'] = np.where((base_treino['DATA_PAGAMENTO'] - base_treino['DATA_VENCIMENTO']).dt.days >= 5, 1, 0)
```
Agora podemos partir para a **análise exploratória**.

### Análise Exploratória




<!--
> ##### CSS Grid Layout Module
>
> This CSS module defines a two-dimensional grid-based layout system, optimized for user interface design. In the grid layout model, the children of a grid container can be positioned into arbitrary slots in a predefined flexible or fixed-size layout grid.
>
> — _W3C_

CSS Grid is a total game changer, IMHO. Compared to the bottomless pit of despair that is the old way, the new way of building a site structure can be done in as little as 5 lines of CSS. Of course, it always takes more than that, but not much. I mean this is really the meat of the deal:

```css
.grid-container {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  grid-template-rows: repeat(3, auto);
}
```

#### What an amazing time to be a web developer. Anyway, I hope you enjoy this "feature" that you'll probably never notice or even see. Maybe that's the best part of a good user interface – the hidden stuff that just works.-->

[^1]: The original article cited here is now updated and maintained by the staff over at CSS-Tricks. Bookmark their version if you want to dive in and learn about CSS Grid: [A Complete Guide to Grid](https://css-tricks.com/snippets/css/complete-guide-grid/)
