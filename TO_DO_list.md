
## TO DO

5. Seprar todos os campos numericos dos campos texto.

6. Pegar a quantidade de valores distintos por campo.
    - Campos com um numero elevado de itens distintos devera ser analisado para verificar se presia de uma classificao por range de valor.
    - Campos com um numero baixo de valores distintos devera ser analisado para verificar se precisa de uma conversao para numerico.
    - Campos com um numero baixo de valores distintos devera ser analisado para verificar se precisa de uma classificacao por range de valor.
  
7. Separa os campos que inicialmente seram utilizados pelos algoritmos.

8. Desenvolvimento
        
    - Criar todos __metodos de classificacao__ que poderao ser utilizados.
    - Criar todos __metodos de agrupamento(clustering)__ que poderao ser utilizados.
    - Criar os __metodos de comparacao__ dos resultados obtidos pelos metodos de __classificacao__.
    - Criar os __metodos de comparacao__ dos resultados obtidos pelos metodos de __regressa__.
    - Criar os __metodos de comparacao__ dos resultados obtidos pelos metodos de __agrupamento__.
    - Criar os __metodos de comparacao__ dos resultasos obtidos pelos __metodos de classificacao x metodos de regressao__.
    - Criar os __metodos de comparacao__ dos resultasos obtidos pelos __metodos de classificacao x metodos de agrupamento__.
    - Criar os __metodos de comparacao__ dos resultasos obtidos pelos __metodos de regressao x metodos de agrupamento__.
    - Criar os __metodos para testar__ os __hyperparameter__ dos metodos de __classificacao__.
    - Criar os __metodos para testar__ os __hyperparameter__ dos metodos de __regressao__.
    - Criar os __metodos para testar__ os __hyperparameter__ dos metodos de __agrupamento__.

## DONE 

1. Identificar campos que devem ser convertidos para numerico.

2. Identificar campos que devem ser classificos por range de valor.

3. Definir qual linguagem sera utilizada. 
    - __Python__
    
4. Definier quais bibliotecas utilizadas e versao das mesmas serao utilizadas.
    - A lista completa estara disponivel no diretorio "4-config/config_linux_env.yml"
    
5.
6.
7.   
8. Desenvolvimento
    
    - Identificar padroes aplicaveis no desenvolvimentos para melhorar consumo de memoria.
    
        Utilizando singleton na leitura do data_set é possivel realizar diferentes metodos de analise no mesmo momento, sem a necessidade de criar varias instancias do data_set na memoria evitando assim um consumo desnecessario de memoria.  Isto possibilita realizar a criacao dos novos campos em um unico objeto em memoria que sera utilizado por todos metodos em paralelo, com isso tambem sera reduzido o tempo de utilizacao de CPU para incluir mais informacoes dentro do data_set.  
        
    - Criar todos __metodos de regressao__ que poderao ser utilizados.
    
        Foi criado o arquivo __"1-scripts/2-ML/regression_model.py"__ contendo todos os modelos que serao utilizados para regressao.
