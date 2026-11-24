# Estudos

**Autor:** Fabio Stein

Repositório de materiais de estudo.

## Conteúdo

| Pasta | Assunto | Arquivos |
|---|---|---|
| [`data-wrangling/`](data-wrangling/) | **Tutorial 1:** Data Wrangling e Visualização de Dados com Python (NumPy, pandas, Matplotlib e seaborn): valores nulos, valores impossíveis, texto inconsistente, duplicados, validação e gráficos, com teoria referenciada | [Markdown](data-wrangling/tutorial_data_wrangling.md) · [Notebook](data-wrangling/tutorial_data_wrangling.ipynb) |
| [`data-wrangling/`](data-wrangling/) | **Tutorial 2:** Outliers e Transformação de Dados: formatos e tipos, detecção de outliers (IQR, Z-score, MAD e Isolation Forest), tratamento (corrigir, marcar, winsorizar, log), amostragem e bootstrap, ordenação e ranking, `replace`/`map`, `cut`/`qcut` e `apply` | [Markdown](data-wrangling/tutorial_data_wrangling_parte2.md) · [Notebook](data-wrangling/tutorial_data_wrangling_parte2.ipynb) |

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
