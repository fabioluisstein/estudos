# Estudos

**Autor:** Fabio Stein

Repositório de materiais de estudo.

## Conteúdo

| Pasta | Assunto | Arquivos |
|---|---|---|
| [`data-wrangling/`](data-wrangling/) | **Tutorial 1:** Data Wrangling e Visualização de Dados com Python (NumPy, pandas, Matplotlib e seaborn): valores nulos, valores impossíveis, texto inconsistente, duplicados, validação e gráficos, com teoria referenciada | [Markdown](data-wrangling/tutorial_data_wrangling.md) · [Notebook](data-wrangling/tutorial_data_wrangling.ipynb) |
| [`data-wrangling/`](data-wrangling/) | **Tutorial 2:** Outliers e Transformação de Dados: formatos e tipos, detecção de outliers (IQR, Z-score, MAD e Isolation Forest), tratamento (corrigir, marcar, winsorizar, log), amostragem e bootstrap, ordenação e ranking, `replace`/`map`, `cut`/`qcut` e `apply` | [Markdown](data-wrangling/tutorial_data_wrangling_parte2.md) · [Notebook](data-wrangling/tutorial_data_wrangling_parte2.ipynb) |
| [`data-wrangling/`](data-wrangling/) | **Tutorial 3:** Valores nulos: técnicas complementares (e duplicados): tipos de nulo do pandas, diagnóstico (MCAR/MAR), `dropna` com critério, preenchimento por tipo de variável, `ffill`/`bfill`/`interpolate`, quando não preencher e qual duplicado manter | [Markdown](data-wrangling/tutorial_data_wrangling_parte3.md) · [Notebook](data-wrangling/tutorial_data_wrangling_parte3.ipynb) |
| [`data-wrangling/`](data-wrangling/) | **Tutorial 4:** Agrupamento e agregação: `groupby`, `agg`, `MultiIndex`/`unstack`, `filter` e `transform`, `pivot_table`, `crosstab`, `melt`, agrupar por tempo e gráficos de grupos | [Markdown](data-wrangling/tutorial_data_wrangling_parte4.md) · [Notebook](data-wrangling/tutorial_data_wrangling_parte4.ipynb) |
| [`data-wrangling/`](data-wrangling/) | **Tutorial 5:** Juntar tabelas: `concat`, `merge` (inner, left, right, outer), cuidados com as chaves, anti-join, `join` pelo índice e checklist | [Markdown](data-wrangling/tutorial_data_wrangling_parte5.md) · [Notebook](data-wrangling/tutorial_data_wrangling_parte5.ipynb) |
| [`data-wrangling/`](data-wrangling/) | **Guia de estudo:** abrir os tutoriais no Google Colab, ordem de estudo, qual tutorial estudar para cada aula, o que é essencial, cola de comandos e glossário espanhol → português | [Guia](data-wrangling/guia_de_estudo.md) |

> **Mais fácil:** abra o [Guia de estudo](data-wrangling/guia_de_estudo.md) e clique em "Abrir no Colab". Os tutoriais rodam no navegador, sem instalar nada.

## Como usar os notebooks

Instale as bibliotecas (uma vez só):

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

Abra o notebook (comece pelo Tutorial 1):

```bash
jupyter notebook data-wrangling/tutorial_data_wrangling.ipynb
jupyter notebook data-wrangling/tutorial_data_wrangling_parte2.ipynb
```
