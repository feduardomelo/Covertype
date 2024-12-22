# 🌲 Classificação de Cobertura Florestal com Dataset Covertype

Este projeto explora a busca por hiperparâmetros ideais para diferentes algoritmos de classificação utilizando o famoso dataset **Covertype**. O objetivo é prever o tipo de cobertura florestal com base em atributos ambientais como elevação, tipo de solo e características climáticas.

## 🚀 Objetivo do Projeto

O objetivo principal foi:
- **Treinar e avaliar** quatro modelos de classificação: Decision Tree, SVM, KNN e MLP.
- **Aplicar Randomized Search** para encontrar os melhores hiperparâmetros.
- **Analisar o desempenho** dos modelos ao longo de múltiplas iterações.

## 📊 Principais Resultados

### Melhores Modelos:
- **KNN:** Melhor desempenho com acurácia de **85.4%**.
- **SVM:** Kernel RBF, alcançando **80.2%** de acurácia.
- **Decision Tree:** Pico de **81.9%** com entropia como critério de divisão.
- **MLP:** Melhor performance com **77.5%**, utilizando arquitetura otimizada.

O gráfico abaixo ilustra a evolução das acurácias ao longo das rodadas de busca de hiperparâmetros:

![Gráfico de Evolução dos Scores](https://github.com/feduardomelo/Covertype/blob/main/Figure_1.png)

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Bibliotecas:**
  - Scikit-Learn
  - Matplotlib
  - Numpy


## 🧠 Aprendizado

- Aprofundamento no uso de **Scikit-Learn**, explorando a construção e avaliação de diferentes algoritmos de classificação.
- Melhoria na capacidade de avaliar e ajustar hiperparâmetros usando **Randomized Search**.
- Aplicação prática de técnicas de pré-processamento de dados e visualização de resultados.

## 📌 Como Reproduzir

1. Clone o repositório:
   ```bash
   git clone https://github.com/feduardomelo/projeto-covertype.git
   
2. Instale as dependências:
   ```bash
   pip install -r numpy sckit-learn matplotlib
   

  ## 📖 Referências
  - Dataset Covertype: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/covertype)
  - Documentação Scikit-Learn: [scikit-learn.org](https://scikit-learn.org/)
  
  ## 📝 Licença
  Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
  

