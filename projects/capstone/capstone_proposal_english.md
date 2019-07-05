# Nanodegree Engenheiro de Machine Learning
## Proposta de Projeto Final - Análise de Sentimentos Aplicada à Previsão de Valores de Ações
Renato Leal de Moura Luz  
05 de julho de 2019

### Domain Background
Sentiment Analysis refers to the use of machine learning to identify the emotional reaction to an event, document or topic [1]. One of the possible applications of sentiment analysis is for predicting stock market movements. The internet is full of sources that represent the public opinion and sentiment about current events. Studies in [2] shows that the aggreagate public mood can be correlated with Dow Jones Industrial Average Index (DJIA).


### Problem Statement

This capstone seeks a model which uses the top daily news headlines from Reddit ( /r/worldnews ) to predict stock market movement. A dataset with 8 years of daily news headlines and their respective DJIAs is available in Kaggle [3]. The stock market movement will be
modeled into a binary classification problem, where:

● 1 when DJIA Adj Close value rose or stayed as the same

● 0 when DJIA Adj Close value decreased .


### CONJUNTO DE DADOS E ENTRADAS

Two channels of data are provided for this dataset:

1. Features: Historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date.


2. Target Variable: Stock data from Dow Jones Industrial Average (DJIA). The index is converted binary values where:

○ 1 when DJIA Adj Close value rose or stayed as the same

○ 0 when DJIA Adj Close value decreased.
  

Three data files are provided on Kaggle in .csv format:

1. RedditNews.csv: two columns The first column is the "date", and second column is the "news headlines". All news are ranked from top to bottom based on how hot they are. Hence, there are 25 lines for each date.

2. DJIA_table.csv: Downloaded directly from Yahoo Finance : check out the web page for more info.

3. Combined_News_DJIA.csv: This is a combined dataset with 27 columns. The first column is "Date", the second is the target variable (DJIA), and the following ones are news headlines ranging from "Top1" to "Top25".


The model will be implemented using only the file Combined_News_DJIA.csv . The range of the data is from 2008-06-08 to 2016-07-01 with a total of 3973 rows . The most recent two years of the dataset (about 20%), from 2015-01-02 to 2016-07-01, is going to be reserved for testing.


### Descrição da solução
_(aprox. 1 parágrafo)_

Nesta seção, descreva claramente uma solução para o problema. A solução deve ser relevante ao assunto do projeto e adequada ao(s) conjunto(s) ou entrada(s) proposto(s). Descreva a solução detalhadamente, de forma que fique claro que o problema é quantificável (a solução pode ser expressa em termos matemáticos ou lógicos), mensurável (a solução pode ser medida por uma métrica e claramente observada) e replicável (a solução pode ser reproduzida e ocorre mais de uma vez).

### Modelo de referência (benchmark)
_(aproximadamente 1-2 parágrafos)_

Nesta seção, forneça os detalhes de um modelo ou resultado de referência que esteja relacionado ao assunto, definição do problema e solução proposta. Idealmente, o resultado ou modelo de referência contextualiza os métodos existentes ou informações conhecidas sobre o assunto e problema propostos, que podem então ser objetivamente comparados à solução. Descreva detalhadamente como o resultado ou modelo de referência é mensurável (pode ser medido por alguma métrica e claramente observado).

### Métricas de avaliação
_(aprox. 1-2 parágrafos)_

Nesta seção, proponha ao menos uma métrica de avaliação que pode ser usada para quantificar o desempenho tanto do modelo de benchmark como do modelo de solução apresentados. A(s) métrica(s) de avaliação proposta(s) deve(m) ser adequada(s), considerando o contexto dos dados, da definição do problema e da solução pretendida. Descreva como a(s) métrica(s) de avaliação pode(m) ser obtida(s) e forneça um exemplo de representação matemática para ela(s) (se aplicável). Métricas de avaliação complexas devem ser claramente definidas e quantificáveis (podem ser expressas em termos matemáticos ou lógicos)

### Design do projeto
_(aprox. 1 página)_

Nesta seção final, sintetize um fluxo de trabalho teórico para obtenção de uma solução para o problema em questão. Discuta detalhadamente quais estratégias você considera utilizar, quais análises de dados podem ser necessárias de antemão e quais algoritmos serão considerados na sua implementação. O fluxo de trabalho e discussão propostos devem estar alinhados com as seções anteriores. Adicionalmente, você poderá incluir pequenas visualizações, pseudocódigo ou diagramas para auxiliar na descrição do design do projeto, mas não é obrigatório. A discussão deve seguir claramente o fluxo de trabalho proposto para o projeto de conclusão.

-----------

**Antes de enviar sua proposta, pergunte-se. . .**

- A proposta que você escreveu segue uma estrutura bem organizada, similar ao modelo de projeto?
- Todas as seções (em especial, **Descrição da solução** e **Design do projeto**) estão escritas de uma forma clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo de seu projeto será capaz de entender sua proposta?
- Você revisou sua proposta de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
