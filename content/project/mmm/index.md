---
title: "Marketing Mix Modelling"
subtitle: "Modelagem bayesiana para predições precisas dos esforços de marketing."
excerpt: "Uma maneira robusta para aferir o resultado dos esforços de marketing."
date: 2025-08-13
author: "Mauricio Pozzebon"
draft: false
tags:
  - Marketing Mix
  - Bayesian
categories:
  - Python
  - Marketing
  - Machine Learning
# layout options: single or single-sidebar
layout: single
---
<style>body {text-align: justify}</style>
<!--{{< here >}}-->

---
À medida que o varejo se torna omnichannel, é fundamental avaliar a contribuição não linear das comunicações de marketing entre seus diversos canais - online e offline - para assim maximizar o retorno sobre o investimento (ROAS[^1]).
 
Nesse contexto estimação bayesiana oferece vantagens significativas em relação a regressão tradicional com transformação logística comum em modelos de marketing. Ela permite identificar os efeitos cumulativos de marketing (adstock) e saturação de investimento (até que ponto continuar investindo gera retorno) conjutamente, além de estimar o orçamento ótimo.

O pacote PyMC (https://www.pymc.io/) facilita esse tipo de modelagem que de outra forma necessitaria de muito poder computacional e algoritmos complexos.Também nos fornece uma série de análises e gráficos explicativos, o que, para gestores de canal, oferece ferramentas robustas para tomada de decisão.

Como exemplo, analisando este dataset sintético identificamos que:

#### 1) O fit do modelo é bom
Percebemos que a estimativa está sempre contendo os reais valores da variável vendas.

<span style="display:block;text-align:center">![Fit do modelo](fit.png)</span>

Os erros de estimativa estão centrados em zero com uma distribuição normal.

<span style="display:block;text-align:center">![errors](errors.png)</span>

#### 2) Os melhores canais são de influencer e instagram, facebook e google_ads desempenham por último
Canais com melhores desempenhos são facilmente reconhecidos.

<span style="display:block;text-align:center">![canais](contribution.png)</span>

#### 3) Instagram e tv tem o maior adstock; Facebook o menor
O efeito cumulativo também pode ser visualizado.

<span style="display:block;text-align:center">![adstock](adstock.png)</span>

#### 4) Facebook e Google Ads saturam mais rápido, Instagram menos
A saturação, ou seja, quando que o canal começa a não responder mais pode ser visualizado nesse gráfico.

<span style="display:block;text-align:center">![saturation](saturation.png)</span>


E o mais impactante, com um orçamento otimizado é possível incrementar o ROAS em ~30%.

<span style="display:block;text-align:center">![optimal](optimal.png)</span>

Por fim, é possível prever o retorno sobre vendas. O gráfico abaixo mostra as vendas provenientes apenas dos efeitos cumulativos e de saturação, sem nenhum investimento alocado.

<span style="display:block;text-align:center">![previsão](prediction.png)</span>

Essa foi apenas uma rápida demonstração do poder preditivo e explicativo desse pacote para modelar e predizer os efeitos das ações de marketing que, no contexto atual de varejo, pode significar milhões a mais ou a menos de faturamento do negócio.

[^1]: Return on Ad Spending.
