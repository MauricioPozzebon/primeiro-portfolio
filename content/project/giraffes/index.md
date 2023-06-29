---
title: "Processamento de Linguagem Natural"
subtitle: "Um modelo ensemble simples de alta precisão."
excerpt: "É possível construir um bom modelo supervisionado de classificação para fins de NLP em poucos passos."
date: 2022-07-15
author: "Mauricio Pozzebon"
draft: false
tags:
  - hugo-site
categories:
  - Python
  - NLP
  - Machine Learning
# layout options: single or single-sidebar
layout: single
---

{{< here >}}

![Tachyons Logo Script](tachyons-logo-script.png)

## [Tachyons](http://tachyons.io) Processamento de linguagem natural (NLP) é um dos tópicos mais quentes em aprendizagem de máquina na atualidade.

---

### NLP e Inteligência Artificial (IA)

Entender o básico de NLP é importante para se aventurar na construção de IA's que entendem a linguagem humana. Pois bem, esse projeto é um modelo de NLP supervisionado que classifica um texto em tópicos pré-estabelecidos e possui **alta precisão.** Graças aos pacotes existentes é possível rodar um modelo python em poucas etapas e código limpo (*clean code*):

**Objetivo:** Classificar qualquer texto entre cinco tópicos (BUSINESS, ENTERTAINMENT, POLITICS, SPORT ou TECH).

A [base de dados](http://mlg.ucd.ie/datasets/bbc.html) consiste em 2225 artigos da BBC contemplando os cinco tópicos entre 2004-2005 já categorizados (por isso um modelo supervisionado).

O *pipeline* segue a estrutura clássica: problema &#x2192; dados &#x2192; preprocessamento &#x2192; modelagem &#x2192; avaliação &#x2192; predição (classificação).


### Código

Primeiramente vamos **importar** todos os pacotes necessários:

```python
import requests
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
```
Para fazer o download diretamente do site, extrair e carregar a base de dados usamos o pacote `requests`:

```python
url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
news = requests.get(url)
open("news.zip", 'wb').write(news.content)

!unzip news.zip

news = load_files('bbc', encoding = 'utf-8', decode_error = 'replace')
```
Para facilitar o treinamento do modelo devemos retirar palavras que não têm "peso" para a classificação como "de", "para", "ao", etc.. No caso usamos uma lista dessas palavras (*stopwords*) em inglês, já que a base está nessa língua. Para isso usamos o pacote `nltk` e fazemos o download da lista que o modelo usará:

```python
nltk.download('stopwords')
```
Explorando a estrutura da base de dados vemos que os tópicos estão divididos em pastas, portanto o pacote `sklearn` deve ser capaz de entender que estamos tratando de um modelo supervisionado:

```python
X = news.data
y = news.target
```
em que `.data` instancia em `X` todos os artigos e `.target` instancia as categorias em `y`.

Vamos utilizar `train_test_split` para dividir a base em treino (70%) e teste (30%).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 93)
```
Até agora os dados continuam estando em forma de texto, é preciso transforma-los em números (vetores) para que o computador consiga "entender" - esse é o segredo do NLP. A transformação se dá pelo módulo `TfidfVectorizer`, aplicando a classe `.fit_transform()` na base de treino e `.transform()` em teste.

```python
vectorizer = TfidfVectorizer(norm = None, stop_words = 'english', max_features = 1000, decode_error = "ignore")

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
```
Agora sim podemos treinar o modelo a partir das bases transformadas.

Uma forma de atingir alta precisão com relativa simplicidade é utilizar modelos *ensemble*: uma combinação de algoritmos, em que os parâmetros estimados nos modelos base são utilizados como parâmatros dentro de um terceiro modelo. No caso faremos um *stack* de três modelos de classificação - dois modelos base (floresta aleatória e Naive Bayes) e um final (regressão logística).

```python
base_model = [('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)), ('nb', MultinomialNB())]

stacked_model = StackingClassifier(estimators = base_model, final_estimator = LogisticRegression(multi_class = 'multinomial', random_state = 30, max_iter = 1000))
```
Finalmente vamos treinar o modelo e na mesma tacada ver a acurácia:

```python
accuracy = stacked_model.fit(np.asarray(X_train_vectors.todense()), y_train).score(np.asarray(X_test_vectors.todense()), y_test)
print(accuracy)
```
Veja que a taxa de acerto atinge **98,2%**!



### Classificando artigos novos

O que realmente estamos interessados é em classificar **novos** artigos. Com um código simples podemos inserir um texto e verificar se o modelo classifica corretamente:

```python
classe = input("texto: ")
data = vectorizer.transform([classe]).toarray()
prediction_output = stacked_model.predict(data)
print(prediction_output)

if prediction_output == 0 :
  print("A classe é: BUSINESS")
if prediction_output == 1 :
  print("A classe é: ENTERTAINMENT")
if prediction_output == 2 :
  print("A classe é: POLITICS")
if prediction_output == 3 :
  print("A classe é: SPORT")
if prediction_output == 4 :
  print("A classe é: TECH")
```
Pronto! O próximo passo seria utilizar uma API para colocar o modelo em produção (Flask ou Django, por exemplo), e deixá-lo em alguma nuvem pública. Discutirei isso num projeto futuro.&#128521; 
