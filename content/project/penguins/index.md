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

{{< figure src="css-grid-cover.png" alt="Traditional right sidebar layout" caption="A visual example of the traditional right sidebar layout" >}}

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

Vemos que a base de treino é pobre em termos de informações individuais. Sendo assim, vamos fundir com as informações cadastrais e "enriquecer" o *dataset* de treino para cada cliente (`ID_CLIENTE`). Em seguida, excluir os dados de cadastro sem correspondência:

```python
base_treino = pd.merge(base_pagamentos_desenvolvimento,base_cadastral,on=['ID_CLIENTE'], how='outer')
base_treino.drop(base_treino.index[77414:], inplace=True)
```
Uma olhada nos dados faltantes agora na base completa:

```python
print(base_treino.isna().sum())
```
Antes de trabalhar os valores nulos podemos excluir as colunas que com certeza **não** usaremos no modelo(`ID_CLIENTE` e `DOMINIO_EMAIL`):

```python
base_treino.drop(columns=['ID_CLIENTE', 'DOMINIO_EMAIL'], inplace = True)
```
A documentação da base indica que `FLAG_PF = X` se pesoa física e `FLAG_PF = NaN` se pessoa jurídica. Portanto:

```python
def flag(x):
    if x == 'X':
        return 'PF'
    else:
        return 'PJ'

base_treino['FLAG'] = base_treino['FLAG_PF'].apply(lambda x: flag(x))
base_treino.drop(columns='FLAG_PF', inplace = True)
```
Agora é necessário tratar `DDD`,`SEGMENTO_INDUSTRIAL` e `PORTE`.

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

#### What an amazing time to be a web developer. Anyway, I hope you enjoy this "feature" that you'll probably never notice or even see. Maybe that's the best part of a good user interface – the hidden stuff that just works.

[^1]: The original article cited here is now updated and maintained by the staff over at CSS-Tricks. Bookmark their version if you want to dive in and learn about CSS Grid: [A Complete Guide to Grid](https://css-tricks.com/snippets/css/complete-guide-grid/)
