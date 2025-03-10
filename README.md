# Projeto: Previsão de Preços de Imóveis

**Descrição:** Projeto de ciência de dados com previsão de preços de imóveis na Califórnia. Os dados utilizados no treinamento da regressão são de 1990 e não devem ser utilizados para fins comerciais nos tempos atuais.

**Autor:** Ivan Luís Duarte  
**LinkedIn:** [linkedin.com/in/ivanluisduarte](https://www.linkedin.com/in/ivanluisduarte/)  
**GitHub:** [github.com/ivanluisduarte](https://github.com/ivanluisduarte)  
**Data de Criação:** 2025-03-10  
**Licença:** MIT  
**Aplicação** [Previsão de Preços de Imóveis](https://california.streamlit.app)  

![Previsão de Preços de Imóveis](./imagens/app.png)

# Modelo de projeto de ciência de dados

Baseado no modelo de [Francisco Bustamante](https://github.com/chicolucio/modelo_projeto_data_science), que foi meu instrutor na base desse trabalho, no treinamento de regressão linear do curso de ciência de dados da Hashtag Treinamentos.

Apesar da idéia inicial ser do curso da Hashtag Treinamentos, todas as decições sobre features, algorítmos e escolha de modelos foi refeita, melhorada e comentada. Um modelo com mais features foi usado ao final. A preparação dos dados para o mapa da interface web e a construção dele ficou mais simples e direta.

Muitas funções foram criadas por mim, tornando esse projeto praticamente um framework para novos projetos de regressão. Uma sequencia clara de uso das funções deixa a construção de pipelines, treinamento e análise de modelos de regressão rápida e intuitiva.

## Organização do projeto

```
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── requirements.txt   <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto (MIT)
├── README.md          <- README principal para desenvolvedores e recrutadores.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── imagens            <- Arquivos de imagens do projeto.
|
├── modelos            <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos
|
├── notebooks          <- Cadernos Jupyter. A convenção de nomenclatura é um número (para ordenação),
│                         as iniciais do criador e uma descrição curta separada por `-`, por exemplo
│                         `02-ild-model_etp7_treinamento_final`.
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py   <- Torna um módulo Python
|      └── auxiliares.py <- Scripts de funções auxiliares que não são de gráficos e nem de modelos.
|      ├── config.py     <- Configurações básicas do projeto como pastas, arquivos e semente para replicar resultados.
|      └── graficos.py   <- Scripts para criar visualizações exploratórias e orientadas a resultados.
|      └── models.py     <- Scripts para treinamento de modelos e avaliação de resultados.
|
└── referencias        <- Dicionários de dados, manuais e todos os outros materiais explicativos.
```
