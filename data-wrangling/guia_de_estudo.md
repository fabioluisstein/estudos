<!-- Autor: Fabio Stein -->

# Guia de estudo: Data Wrangling com Python

**Autor:** Fabio Stein

Este guia é um **material de apoio** para estudar com os cinco tutoriais desta pasta. Ele diz **por onde começar**, **qual parte estudar para cada aula** e **o que dá para pular na primeira leitura**. Os tutoriais em si não mudam.

---

## 1. Abrir os tutoriais sem instalar nada

Clique no botão de cada tutorial. Ele abre no **Google Colab**, direto no navegador, com Python e todas as bibliotecas já instalados. Só é preciso uma conta Google.

| Tutorial | Assunto | Abrir |
|---|---|---|
| 1 | Valores nulos, valores impossíveis, texto bagunçado, duplicados e gráficos | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabioluisstein/estudos/blob/main/data-wrangling/tutorial_data_wrangling.ipynb) |
| 2 | Outliers, formatos e transformação de dados | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabioluisstein/estudos/blob/main/data-wrangling/tutorial_data_wrangling_parte2.ipynb) |
| 3 | Valores nulos: técnicas complementares (e duplicados) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabioluisstein/estudos/blob/main/data-wrangling/tutorial_data_wrangling_parte3.ipynb) |
| 4 | Agrupamento e agregação de dados | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabioluisstein/estudos/blob/main/data-wrangling/tutorial_data_wrangling_parte4.ipynb) |
| 5 | Juntar tabelas (concat, merge e join) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabioluisstein/estudos/blob/main/data-wrangling/tutorial_data_wrangling_parte5.ipynb) |

**Dicas do Colab**
- Execute as células **em ordem**, de cima para baixo (`Shift + Enter`). Se der erro estranho, use **Ambiente de execução → Reiniciar e executar tudo**.
- O que você muda no Colab **não altera** o tutorial original. Para guardar suas anotações e respostas: **Arquivo → Salvar uma cópia no Drive**.
- Os arquivos CSV que o tutorial cria somem quando o Colab é fechado. Basta rodar o tutorial de novo.

Prefere usar no seu computador? Veja a seção "Como usar os notebooks" no [README](../README.md).

---

## 2. Como estudar cada tutorial

1. **Primeira leitura:** leia o texto, o **🔍 Passo a passo** e rode o código. **Pule os quadros 📚 Teoria**: eles explicam o "porquê" e podem esperar.
2. **Mexa no código:** troque um número, um nome de coluna, um parâmetro. Veja o que muda. Errar aqui é aprender.
3. **Faça a atividade da aula** correspondente (tabela da seção 3).
4. **Faça os exercícios** do fim do tutorial.
5. **Segunda leitura:** agora sim, leia os quadros 📚 Teoria. Eles vão fazer muito mais sentido.

> Ordem recomendada: **1 → 3 → 2 → 4 → 5**. O Tutorial 3 completa o 1 (os dois falam de nulos), e o 4 vem antes do 5 porque o último usa agrupamento.

---

## 3. Qual tutorial estudar para cada aula

Os nomes de arquivo são os do material da disciplina.

### Módulo 06: Valores nulos

| Aula | Tema | Onde estudar |
|---|---|---|
| `TOPD2026-2_21.pdf` | Por que os dados vêm "sujos": nulos, duplicados, outliers e formatos; o fluxo de limpeza | Tutorial 1: Conceitos fundamentais e Partes 2 a 7 · Tutorial 2: Parte 2 (formatos) |
| `TOPD2026-2_22` | O que são nulos; apagar ou preencher; preenchimento por tipo de variável | Tutorial 1: Partes 3 e 6 · Tutorial 3: Partes 1, 3, 4 e 6 |
| `TOPD2026-2_23` | `isnull`, `dropna`, `fillna`, preencher para frente e para trás | Tutorial 1: Partes 3 e 6 · Tutorial 3: Partes 3 e 5 |
| `TOPD2026-2_24` | Roteiro de análise de nulos: detectar, achar padrões, medir o impacto | Tutorial 3: Partes 2, 3 e 8 |
| `TOPD2026-2_25` | Exercício integrador de nulos | Use o "Resumo do fluxo" dos Tutoriais 1 e 3 como roteiro |

### Módulo 07: Data Wrangling 2

| Aula | Tema | Onde estudar |
|---|---|---|
| `TOPD2026-2_26` | Outliers: detecção visual, IQR, Z-score, MAD, Isolation Forest e tratamento | Tutorial 2: Partes 3 a 6 |
| `TOPD2026-2_27` | Amostragem, permutação e ordenação | Tutorial 2: Partes 7 e 8 |
| `TOPD2026-2_28` | Duplicados, substituição de valores, faixas (`cut`/`qcut`) e funções (`apply`) | Tutorial 1: Parte 5.2 · Tutorial 3: Parte 7 · Tutorial 2: Partes 9 a 11 |

### Módulo 08: Agrupamento e agregação

| Aula | Tema | Onde estudar |
|---|---|---|
| `TOPD2026-2_29` | `groupby` em Series e DataFrames; tabelas dinâmicas | Tutorial 4: Partes 2 e 6 |
| `TOPD2026-2_30` | `groupby`, várias funções de agregação, várias colunas, filtros, `transform` e percorrer linhas | Tutorial 4: Partes 2 a 5 e 8 |

### Módulo 09: Junção de tabelas

| Aula | Tema | Onde estudar |
|---|---|---|
| `TOPD2026-2_31` | `concat`, `merge` e `join` | Tutorial 5: Partes 2, 3, 4 e 6 |
| `TOPD2026-2_32` | Exercício de junção: registros sem par | Tutorial 5: Partes 3 e 5 |
| `TOPD2026-2_33` | `left`, `right`, `inner` e `outer` passo a passo | Tutorial 5: Partes 3 e 5 |

---

## 4. Essencial × pode esperar (Tutoriais 1 e 2)

Os Tutoriais 3, 4 e 5 já marcam a teoria como **(opcional)** no título. Para os dois primeiros, use esta lista.

### Tutorial 1

| Parte | Essencial | Pode esperar |
|---|---|---|
| Conceitos fundamentais | o que é data wrangling e a tabela de dimensões da qualidade | *tidy data* |
| 0 a 2 | tudo (carregar e olhar os dados) | quadros de teoria |
| 3 Valores nulos | 3.1 a 3.4 | mecanismos de Rubin (veja de novo no Tutorial 3) |
| 4 Valores impossíveis | 4.1 a 4.3 | regra de Tukey (aparece de novo no Tutorial 2) |
| 5 Texto e duplicados | tudo | |
| 6 Apagar ou preencher | 6.1 a 6.3 | "por que a mediana é robusta" e imputação condicional |
| 7 Validar | tudo | |
| 8 Visualização | 8.1 a 8.5 | Teorias 1 a 4, mapa de calor e painel (8.6 e 8.7) |

### Tutorial 2

| Parte | Essencial | Pode esperar |
|---|---|---|
| 1 e 2 Dados e formatos | tudo | histórias de acidentes com formatos |
| 3 Olhar antes de calcular | boxplot e 3.1 (estrago na média) | 3.2 (reta de tendência) |
| 4 IQR, Z-score e MAD | 4.1, 4.2, 4.3 e 4.5 | fórmula de Shiffler e 4.4 |
| 5 Isolation Forest | 5.1 | 5.2 |
| 6 Tratar outliers | 6.1, 6.2 e 6.3 | 6.4 (winsorização) e 6.5 (log) |
| 7 Amostragem | 7.1, 7.2 e 7.4 | 7.3 (bootstrap) e 7.5 (permutação), mas a aula 27 pede os dois nos exercícios |
| 8 Ordenação | 8.1, 8.3 e 8.5 | 8.4 (tipos de empate) |
| 9 a 11 Substituir, faixas e funções | tudo | comparação de tempo em 11.4 |
| 12 Validar | tudo | |

---

## 5. Cola de comandos (pandas)

| Quero... | Comando |
|---|---|
| ver o começo da tabela | `df.head()` |
| ver tipos e nulos | `df.info()` |
| resumo estatístico | `df.describe()` |
| contar valores de uma coluna | `df["col"].value_counts()` |
| contar nulos por coluna | `df.isna().sum()` |
| porcentagem de nulos | `df.isna().mean() * 100` |
| apagar linhas com nulo | `df.dropna()` / `df.dropna(subset=["col"])` |
| apagar colunas com nulo | `df.dropna(axis=1)` |
| preencher nulos | `df["col"].fillna(df["col"].median())` |
| repetir o valor anterior / seguinte | `df["col"].ffill()` / `df["col"].bfill()` |
| interpolar | `df["col"].interpolate()` |
| texto para número | `pd.to_numeric(df["col"], errors="coerce")` |
| texto para data | `pd.to_datetime(df["col"], format="%d/%m/%Y", errors="coerce")` |
| padronizar texto | `df["col"].str.strip().str.lower()` |
| achar / apagar duplicados | `df.duplicated()` / `df.drop_duplicates(subset=[...], keep="first")` |
| limites do IQR | `q1, q3 = df["col"].quantile([0.25, 0.75])` |
| limitar valores | `df["col"].clip(minimo, maximo)` |
| amostra | `df.sample(n=10, random_state=1)` / `df.sample(frac=0.2)` |
| embaralhar | `df.sample(frac=1).reset_index(drop=True)` |
| ordenar | `df.sort_values(["a", "b"], ascending=[True, False])` |
| maiores valores | `df.nlargest(5, "col")` |
| trocar valores | `df["col"].replace({"a": "b"})` / `df["col"].map(dicionario)` |
| se/senão | `np.where(condicao, "sim", "não")` |
| faixas | `pd.cut(df["col"], bins=[...], labels=[...])` / `pd.qcut(df["col"], q=4)` |
| aplicar função | `df["col"].apply(funcao)` / `df.apply(funcao, axis=1)` |
| agrupar e resumir | `df.groupby("grupo")["col"].agg(["mean", "sum"])` |
| resultado do grupo em cada linha | `df.groupby("grupo")["col"].transform("mean")` |
| tabela dinâmica | `df.pivot_table(index="a", columns="b", values="c", aggfunc="mean")` |
| empilhar tabelas | `pd.concat([df1, df2], ignore_index=True)` |
| juntar por chave | `pd.merge(esq, dir, on="id", how="left")` |
| ver quem ficou sem par | `pd.merge(esq, dir, on="id", how="outer", indicator=True)` |
| conferir uma regra | `assert (df["idade"] >= 0).all()` |

---

## 6. Glossário espanhol → português

As aulas da disciplina são em espanhol, e os tutoriais em português. Estes são os termos que mais aparecem.

| Espanhol (aula) | Português (tutoriais) | Observação |
|---|---|---|
| valores faltantes / nulos | valores nulos / faltantes | |
| datos atípicos, *outliers* | outliers, valores atípicos | |
| imputación | imputação, preenchimento | |
| eliminación por lista (*listwise deletion*) | apagar as linhas com nulo | |
| **fila** | **linha** | ⚠️ falso amigo: em espanhol, *fila* é linha da tabela |
| columna | coluna | |
| promedio / media | média | |
| desviación estándar | desvio-padrão | |
| rango intercuartílico (RIC) | intervalo interquartil (IQR) | |
| **sesgo** | **viés** ou **assimetria** | ⚠️ *distribución sesgada* = distribuição **assimétrica**; *sesgo* numa amostra = **viés** |
| muestreo (estratificado) | amostragem (estratificada) | |
| con reemplazo | com reposição | |
| ordenamiento | ordenação | |
| reemplazo de valores | substituição de valores | |
| discretización, *binning* | discretização, faixas | |
| agrupamiento / agregación | agrupamento / agregação | |
| tabla dinámica | tabela dinâmica | |
| unión / combinación | junção | |
| llave / clave | chave | |
| ingreso | renda | |
| encuesta | pesquisa, questionário | |
| hijos | filhos | |
