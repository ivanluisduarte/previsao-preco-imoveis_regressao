from IPython.display import display
import os
import pandas as pd


from sklearn.base import (
    RegressorMixin, # Classe de regressão que deve implementar fit e predict
    BaseEstimator, # Classe de transformação que deve implementar fit e transform
)
from sklearn.compose import TransformedTargetRegressor # para transformar a variável alvo/target
from sklearn.metrics import ( # https://scikit-learn.org/stable/api/sklearn.metrics.html#regression-metrics
    d2_absolute_error_score,        # D² regression score function, fraction of absolute error explained.
    d2_pinball_score,               # D² regression score function, fraction of pinball loss explained.
    d2_tweedie_score,               # D² regression score function, fraction of Tweedie deviance explained.
    explained_variance_score,       # Explained variance regression score function.
    max_error,                      # The max_error metric calculates the maximum residual error.
    mean_absolute_error,            # Mean absolute error regression loss.
    mean_absolute_percentage_error, # Mean absolute percentage error (MAPE) regression loss.
    mean_gamma_deviance,            # Mean Gamma deviance regression loss.
    mean_pinball_loss,              # Pinball loss for quantile regression.
    mean_poisson_deviance,          # Mean Poisson deviance regression loss.
    mean_squared_error,             # Mean squared error regression loss.
    mean_squared_log_error,         # Mean squared logarithmic error regression loss.
    mean_tweedie_deviance,          # Mean Tweedie deviance regression loss.
    median_absolute_error,          # Median absolute error regression loss.
    r2_score,                       # R² (coefficient of determination) regression score function.
    root_mean_squared_error,        # Root mean squared error regression loss.
    root_mean_squared_log_error,    # Root mean squared logarithmic error regression loss.
)
from sklearn.model_selection import (
    cross_validate, # validação cruzada para regressão
    GridSearchCV,
    KFold, # validação cruzada (k-fold) para regressão
)
from sklearn.pipeline import Pipeline # constroer pipeline com pré-processamento e regressor

from .config import RANDOM_STATE
from .auxiliares import fnc_exibir_dataframe_resultados
from .graficos import PALETTE_TEMPERATURA










##########################################################################
# Função para treinar um único modelo de regressão e avaliar o desempenho.
##########################################################################
def fnc_treinar_e_validar_um_modelo_regressao(
    modelo: object,
    X_train: pd.DataFrame | pd.Series, # Dados preditores para treinamento
    y_train: pd.DataFrame | pd.Series, # Dados preditos para treinamento
    X_test: pd.DataFrame | pd.Series, # Dados preditores para teste
    y_test: pd.DataFrame | pd.Series, # Dados preditos para teste
    X_validation: pd.DataFrame | pd.Series = None, # Dados preditores para validação
    y_validation: pd.DataFrame | pd.Series = None, # Dados preditos para validação
    metricas: list | str | tuple= ( # string ou coleção com as metrícas a avaliar
        'adjusted_r2_score',              # R² ajustado
        'd2_absolute_error_score',        # D² regression score function, fraction of absolute error explained.
        'd2_pinball_score',               # D² regression score function, fraction of pinball loss explained.
        'd2_tweedie_score',               # D² regression score function, fraction of Tweedie deviance explained.
        'explained_variance_score',       # Explained variance regression score function.
        'max_error',                      # The max_error metric calculates the maximum residual error.
        'mean_absolute_error',            # Mean absolute error regression loss.
        'mean_absolute_percentage_error', # Mean absolute percentage error (MAPE) regression loss.
        'mean_gamma_deviance',            # Mean Gamma deviance regression loss.
        'mean_pinball_loss',              # Pinball loss for quantile regression.
        'mean_poisson_deviance',          # Mean Poisson deviance regression loss.
        'mean_squared_error',             # Mean squared error regression loss.
        'mean_squared_log_error',         # Mean squared logarithmic error regression loss.
        'mean_tweedie_deviance',          # Mean Tweedie deviance regression loss.
        'median_absolute_error',          # Median absolute error regression loss.
        'r2_score',                       # R² (coefficient of determination) regression score function.
        'root_mean_squared_error',        # Root mean squared error regression loss.
        'root_mean_squared_log_error',    # Root mean squared logarithmic error regression loss.
    ),
) -> pd.DataFrame:
    '''
        Treina e valida um modelo de regressão, retornando métricas de desempenho.

        Parameters
        ----------
        modelo : object
            Modelo de regressão a ser treinado (deve ter métodos fit e predict).
        X_train : pd.DataFrame ou pd.Series
            Dados preditores para treinamento.
        y_train : pd.DataFrame ou pd.Series
            Variável alvo para treinamento.
        X_test : pd.DataFrame ou pd.Series
            Dados preditores para teste.
        y_test : pd.DataFrame ou pd.Series
            Variável alvo para teste.
        X_validation : pd.DataFrame ou pd.Series, opcional
            Dados preditores para validação (default=None).
        y_validation : pd.DataFrame ou pd.Series, opcional
            Variável alvo para validação (default=None).
        metricas : list, str ou tuple, opcional
            Métricas de avaliação a serem calculadas. Default inclui todas as métricas disponíveis.

        Returns
        -------
        pd.DataFrame
            DataFrame contendo as métricas calculadas para treino, teste e (se fornecidos) validação.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression
        >>> modelo = LinearRegression()
        >>> resultados = fnc_treinar_e_validar_um_modelo_regressao(modelo, X_treino, y_treino, X_teste, y_teste)
    '''
    
    ###########################################################################################
    # SUB-FUNÇÃO que calcula as métricas de avaliação especificadas para as previsões do modelo
    ###########################################################################################
    def subfnc_metricas(
        y_true: pd.DataFrame | pd.Series,
        y_pred: pd.DataFrame | pd.Series,
    ) -> pd.Series: # Uma coluna que pode ser inserida num dataframe. O índice é o nome da metrica.
        '''Calcula as métricas de avaliação especificadas para as previsões do modelo.'''
        # Dicionário para armazenar os resultados das métricas
        dct_resultados = {}
        
        # Avaliação das métricas solicitadas
        if 'adjusted_r2_score' in metricas:
            # Calcula o R² ajustado considerando o número de preditores
            dct_resultados['adjusted_r2_score'] = 1.0 - (
                (1.0 - r2_score(y_true, y_pred)) *
                (len(y_true) - 1.0) /
                (len(y_true) - X_train.shape[1] - 1.0)
            )
        if 'd2_absolute_error_score' in metricas:
            dct_resultados['d2_absolute_error_score'] = d2_absolute_error_score(y_true, y_pred)
        if 'd2_pinball_score' in metricas: 
            dct_resultados['d2_pinball_score'] = d2_pinball_score(y_true, y_pred)
        if 'd2_tweedie_score' in metricas: 
            dct_resultados['d2_tweedie_score'] = d2_tweedie_score(y_true, y_pred)
        if 'explained_variance_score' in metricas: 
            dct_resultados['explained_variance_score'] = explained_variance_score(y_true, y_pred)
        if 'max_error' in metricas: 
            dct_resultados['max_error'] = max_error(y_true, y_pred)
        if 'mean_absolute_error' in metricas: 
            dct_resultados['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
        if 'mean_absolute_percentage_error' in metricas: 
            dct_resultados['mean_absolute_percentage_error'] = mean_absolute_percentage_error(y_true, y_pred)
        if 'mean_gamma_deviance' in metricas: 
            dct_resultados['mean_gamma_deviance'] = mean_gamma_deviance(y_true, y_pred)
        if 'mean_pinball_loss' in metricas: 
            dct_resultados['mean_pinball_loss'] = mean_pinball_loss(y_true, y_pred)
        if 'mean_poisson_deviance' in metricas: 
            dct_resultados['mean_poisson_deviance'] = mean_poisson_deviance(y_true, y_pred)
        if 'mean_squared_error' in metricas: 
            dct_resultados['mean_squared_error'] = mean_squared_error(y_true, y_pred)
        if 'mean_squared_log_error' in metricas: 
            dct_resultados['mean_squared_log_error'] = mean_squared_log_error(y_true, y_pred)
        if 'mean_tweedie_deviance' in metricas: 
            dct_resultados['mean_tweedie_deviance'] = mean_tweedie_deviance(y_true, y_pred)
        if 'median_absolute_error' in metricas: 
            dct_resultados['median_absolute_error'] = median_absolute_error(y_true, y_pred)
        if 'r2_score' in metricas: 
            dct_resultados['r2_score'] = r2_score(y_true, y_pred)
        if 'root_mean_squared_error' in metricas: 
            dct_resultados['root_mean_squared_error'] = root_mean_squared_error(y_true, y_pred)
        if 'root_mean_squared_log_error' in metricas: 
            dct_resultados['root_mean_squared_log_error'] = root_mean_squared_log_error(y_true, y_pred)
        
        # Retorna uma Series pandas com os resultados
        return pd.Series(data=dct_resultados)

    # Treina o modelo usando os dados de treinamento
    modelo.fit(X_train, y_train)

    # Cria um DataFrame para armazenar os resultados
    df_resultados = pd.DataFrame()
    
    # Calcula métricas para os dados de treino
    df_resultados['Treino'] = subfnc_metricas(y_train, modelo.predict(X_train))
    # Calcula métricas para os dados de teste
    df_resultados['Teste'] = subfnc_metricas(y_test, modelo.predict(X_test))
    
    # Se dados de validação forem fornecidos, calcula métricas para eles
    if X_validation is not None and y_validation is not None:
        df_resultados['Validação'] = subfnc_metricas(y_validation, modelo.predict(X_validation))

    return df_resultados











################################################################################
# Função que retorna um pipeline de modelo pronto para treinamento
################################################################################
def fnc_construir_pipeline_modelo_regressao(
    # PARAMETRO OBRIGATORIO
    regressor: RegressorMixin,  # Classe de regressão que deve implementar fit e predict
    # PARAMETROS OPCIONAIS
    preprocessor: BaseEstimator = None,  # Classe de pré-processamento, deve implementar fit e transform
    target_transformer: BaseEstimator = None,  # Classe de transformação de alvo, deve implementar fit e transform

    nome_saida_preprocessor: str = 'preprocessor',  # Nome que aparecerá na saída identificando o pré-processador
    nome_saida_regressor: str = 'regressor',  # Nome que aparecerá na saída identificando o regressor
) -> TransformedTargetRegressor:  # Modelo configurado pronto para treinamento
    '''
    Função que constrói um pipeline de modelo de regressão pronto para treinamento.

    Args:
        regressor (RegressorMixin): Classe de regressão que deve implementar os métodos fit e predict.
        preprocessor (BaseEstimator, optional): Classe de pré-processamento de features que deve implementar fit e transform. Padrão é None.
        target_transformer (BaseEstimator, optional): Classe de transformação de alvo que deve implementar fit e transform. Padrão é None.
        nome_saida_preprocessor (str, optional): Nome que aparecerá na saída identificando o pré-processador. Padrão é 'preprocessor'.
        nome_saida_regressor (str, optional): Nome que aparecerá na saída identificando o regressor. Padrão é 'regressor'.

    Returns:
        TransformedTargetRegressor: Modelo configurado pronto para treinamento, podendo incluir transformação de alvo.

    Examples:
        >>> model = fnc_construir_pipeline_modelo_regressao(regressor)
        >>> model.fit(X, y)
    '''

    # Construindo o pipeline
    if preprocessor is None:  # Apenas regressor
        pipeline = Pipeline([(nome_saida_regressor, regressor)])
    else:  # Com pré-processador e regressor
        pipeline = Pipeline([(nome_saida_preprocessor, preprocessor), (nome_saida_regressor, regressor)])

    if target_transformer is None:  # Sem transformação de alvo
        model = pipeline
    else:  # Com transformação de alvo
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )

    return model






################################################################################
# função para planejar modelos de regressão validação cruzada e grid search
################################################################################
def fnc_grid_search_cv_regressor(
    # PARAMETROS OBRIGATÓRIOS
    regressor: RegressorMixin,  # Classe de regressão que deve implementar fit e predict
    param_grid: dict|list, # dicionário, ou lista de dicionários contendo os hiperparametros a testar. A chave precisa respeitar a etapa/parametro.
    # PARAMETROS OPCIONAIS
    preprocessor: BaseEstimator = None,  # Classe de pré-processamento, deve implementar fit e transform
    target_transformer: BaseEstimator = None,  # Classe de transformação de alvo, deve implementar fit e transform
    nome_saida_preprocessor: str= 'preprocessor',# nome/chave que aparecerá na saída identificando o preprocessor
    nome_saida_regressor: str= 'regressor', # nome/chave que aparecerá na saída identificando o regressor
    n_splits: int= 5, # numero de pedaços e vezes em que a base será divida para treino
    shuffle: bool= True, # True indica que os splits devem ser pegos formados de forma aleatória e False indica sequência original
    random_state: int= RANDOM_STATE, # semente para reprodução dos resultados
    scoring: list|tuple= ( # lista/tupla com os algorítmos de avaliação de modelos de regressão
        # 'd2_absolute_error_score', # melhor = 1.0 - dummy seria zero - pode ser negativo -> método metrics.d2_absolute_error_score
        # 'explained_variance', #  melhor = 1.0 -> método metrics.explained_variance_score
        # 'neg_max_error', # melhor = 0.0 -> método metrics.max_error
        'neg_mean_absolute_error', # melhor = 0.0 -> método metrics.mean_absolute_error
        # 'neg_mean_absolute_percentage_error', # melhor = 0.0 -> metrics.mean_absolute_percentage_error
        # 'neg_mean_gamma_deviance', # melhor = 0.0 ->  metrics.mean_gamma_deviance
        # 'neg_mean_poisson_deviance', # melhor = 0.0 -> metrics.mean_poisson_deviance
        # 'neg_mean_squared_error', # melhor = 0.0 -> metrics.mean_squared_error
        # 'neg_mean_squared_log_error', # melhor = 0.0 -> metrics.mean_squared_log_error
        # 'neg_median_absolute_error', # melhor = 0.0 -> metrics.median_absolute_error
        'neg_root_mean_squared_error', # melhor = 0.0 -> metrics.root_mean_squared_error
        # 'neg_root_mean_squared_log_error', # melhor = 0.0 -> metrics.root_mean_squared_log_error
        'r2', # melhor = 1.0 -> metrics.r2_score
    ),
    refit: bool|str= 'neg_root_mean_squared_error', # algorítmos de avaliação de modelos de regressão que será usado como avaliador do melhor modelo
    n_jobs: int= -2, # número de job/processadores simultaneos. Se None, 1 job por vez apenas, se -1 usa todos os processadores disponíveis, se -2 usa todos -1 (use se a máquina estiver travando usando todos)
    return_train_score: bool= False, # True se quiser dados de treinamento na saída
    verbose: int= 1 # nível de log que deseja ver - 1, 2 ou 3
) -> GridSearchCV: # retorna um plano de execução pronto para treinar com fit() passando X e y
    '''
        Função para planejar um modelo de regressão utilizando validação cruzada e grid search para ajuste de hiperparâmetros.

        Args:
            regressor (RegressorMixin): Classe de regressão que deve implementar os métodos fit e predict.
            param_grid (dict | list): Dicionário ou lista de dicionários contendo os hiperparâmetros a testar. As chaves devem respeitar a estrutura do pipeline.
            preprocessor (BaseEstimator, optional): Classe de pré-processamento de features que deve implementar os métodos fit e transform. Padrão é None.
            target_transformer (BaseEstimator, optional): Classe de transformação de alvo que deve implementar os métodos fit e transform. Padrão é None.
            nome_saida_preprocessor (str, optional): Nome que aparecerá na saída identificando o pré-processador. Padrão é 'preprocessor'.
            nome_saida_regressor (str, optional): Nome que aparecerá na saída identificando o regressor. Padrão é 'regressor'.
            n_splits (int, optional): Número de divisões (folds) para a validação cruzada. Padrão é 5.
            shuffle (bool, optional): Se True, as divisões serão feitas de forma aleatória. Se False, a sequência original será mantida. Padrão é True.
            random_state (int, optional): Semente para a geração aleatória, permitindo a reprodução dos resultados. Padrão é RANDOM_STATE.
            scoring (list | tuple, optional): Lista ou tupla com os métodos de avaliação para os modelos de regressão. Padrão inclui 'neg_mean_absolute_error', 'neg_root_mean_squared_error' e 'r2'.
            refit (bool | str, optional): Algoritmo de avaliação a ser utilizado para reajustar o modelo após a busca de hiperparâmetros. Padrão é 'neg_root_mean_squared_error'.
            n_jobs (int, optional): Número de processos simultâneos a serem utilizados. Padrão é -1 (todos os processadores disponíveis).
            return_train_score (bool, optional): Se True, inclui os dados de treinamento na saída. Padrão é False.
            verbose (int, optional): Nível de detalhe na saída de logs. Aceita 0, 1, 2 ou 3. Padrão é 1.

        Returns:
            GridSearchCV: Retorna um objeto GridSearchCV configurado e pronto para ser ajustado com os dados X e y.

        Examples:
            >>> grid_search = fnc_grid_search_cv_regressor(regressor, param_grid)
            Plano de execução pronto para treinar.
   '''
    
    # <<< Implementação da função >>>


    # criando um pipeline de modelo pronto para treinamento
    model = fnc_construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer, nome_saida_preprocessor, nome_saida_regressor
    )

    # configurando a validação cruzada
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # criando uma plano de mudança de hiperparametros para utilizar no modelo criado acima
    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        verbose=verbose,
    )

    # retornando um plano de execução, o FIT ainda precisa ser feito enviado os dados X e y
    return grid_search





#########################################################################################################################
# função para organizar os resultados da validação cruzada feita pela função treinar_e_validar_modelo_regressao() acima #
#########################################################################################################################
def fnc_organiza_resultados(
    dct_resultados: dict,  # Dicionário com os resultados da validação cruzada
) -> pd.DataFrame:  # DataFrame com os resultados organizados
    '''
    Função para organizar os resultados da validação cruzada em um DataFrame.

    Args:
        dct_resultados (dict): Dicionário contendo os resultados da validação cruzada. Cada chave deve representar um modelo e cada valor deve ser outro dicionário com os resultados correspondentes, incluindo 'fit_time' e 'score_time'.

    Returns:
        pd.DataFrame: Um DataFrame contendo os resultados organizados da validação cruzada, onde cada linha representa uma iteração de validação e as colunas incluem os resultados e o tempo total de treinamento e validação.

    Examples:
        >>> df_resultados = fnc_organiza_resultados(dct_resultados)
        >>> print(df_resultados.head())
    
    Notes:
        O DataFrame resultante terá uma coluna 'model' com os nomes dos modelos e as colunas de resultados expandidas, além de uma coluna adicional 'time_seconds' que representa o tempo total gasto em cada iteração de validação cruzada.
    '''

    # Calcula o tempo total de treinamento e validação para cada modelo
    for chave in dct_resultados:
        dct_resultados[chave]['time_seconds'] = (
            dct_resultados[chave]['fit_time'] + dct_resultados[chave]['score_time']
        )

    # Cria um DataFrame a partir dos resultados da validação cruzada
    df_resultados = (
        pd.DataFrame(dct_resultados).T.reset_index().rename(columns={'index': 'model'})
    )

    # Expande o DataFrame para que cada linha corresponda a uma iteração de validação cruzada
    df_resultados = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)
    print(df_resultados.columns)

    # Converte as colunas para o tipo numérico, ignorando erros de conversão
    for coluna in df_resultados.columns.difference(['model']):
        # reduzindo o tamanho das variáveis numericas
        # todas se tornam o menor float possível de acordo com o conteúdo
        # apenas as que podem ser inteiras são transformadas para o menor inteiro possível
        # apenas as que podem ser naturais são transformadas para o menor natural possível
        for tipo_base in ('float', 'signed', 'unsigned'):
            try:
                df_resultados[coluna] = pd.to_numeric(df_resultados[coluna], downcast=tipo_base)
            except ValueError:
                pass


    return df_resultados











########################################################################################
# Função para treinar modelos de regressão com validação cruzada e avaliar o desempenho.
########################################################################################
def fnc_treinar_e_validar_modelos_regressao(
    X: pd.DataFrame | pd.Series,  # Dados preditores
    y: pd.DataFrame | pd.Series,  # Variável predita
    regressors: dict,  # Dicionário de modelos com pré-processadores e regressores
    nome_saida_preprocessor: str = 'preprocessor',  # Nome para identificar o pré-processador na saída
    nome_saida_regressor: str = 'regressor',  # Nome para identificar o regressor na saída
    n_splits: int = 5,  # Número de folds na validação cruzada
    shuffle: bool = True,  # Se True, embaralha os dados antes da divisão
    random_state: int = RANDOM_STATE,  # Semente para reprodutibilidade
    scoring: list|tuple= ( # lista/tupla com métricas de avaliação
        # 'd2_absolute_error_score', # melhor = 1.0 - dummy seria zero - pode ser negativo -> método metrics.d2_absolute_error_score
        # 'explained_variance', #  melhor = 1.0 -> método metrics.explained_variance_score
        # 'neg_max_error', # melhor = 0.0 -> método metrics.max_error
        'neg_mean_absolute_error', # melhor = 0.0 -> método metrics.mean_absolute_error
        # 'neg_mean_absolute_percentage_error', # melhor = 0.0 -> metrics.mean_absolute_percentage_error
        # 'neg_mean_gamma_deviance', # melhor = 0.0 ->  metrics.mean_gamma_deviance
        # 'neg_mean_poisson_deviance', # melhor = 0.0 -> metrics.mean_poisson_deviance
        # 'neg_mean_squared_error', # melhor = 0.0 -> metrics.mean_squared_error
        # 'neg_mean_squared_log_error', # melhor = 0.0 -> metrics.mean_squared_log_error
        # 'neg_median_absolute_error', # melhor = 0.0 -> metrics.median_absolute_error
        'neg_root_mean_squared_error', # melhor = 0.0 -> metrics.root_mean_squared_error
        # 'neg_root_mean_squared_log_error', # melhor = 0.0 -> metrics.root_mean_squared_log_error
        'r2', # melhor = 1.0 -> metrics.r2_score
    ),
) -> pd.DataFrame:  # DataFrame com resultados
    '''
    Função para treinar modelos de regressão com validação cruzada e avaliar o desempenho.

    Args:
        X: Dados preditores.
        y: Variável alvo.
        regressors: Dicionário contendo modelos, pré-processadores e transformadores de alvo.
            Exemplo:
            regressors = {
                'LinearRegression': {
                    'preprocessor': ColumnTransformer(...),  # Opcional
                    'regressor': LinearRegression(),  # Obrigatório
                    'target_transformer': PowerTransformer(),  # Opcional
                },
                'Ridge': { ... } # Outro modelo
            }
        nome_saida_preprocessor (str, opcional): Nome para identificar o pré-processador na saída. Padrão: 'preprocessor'.
        nome_saida_regressor (str, opcional): Nome para identificar o regressor na saída. Padrão: 'regressor'.
        n_splits (int, opcional): Número de folds na validação cruzada. Padrão: 5.
        shuffle (bool, opcional): Se True, embaralha os dados antes da divisão. Padrão: True.
        random_state (int, opcional): Semente para reprodutibilidade. Padrão: RANDOM_STATE.
        flg_mostrar_dataframe (bool, opcional): Se True, exibe o DataFrame de resultados. Padrão: True.
        paleta_de_cores (str, opcional): Paleta de cores para o DataFrame. Padrão: PALETTE_TEMPERATURA.
        precisao (int, opcional): Precisão decimal para o DataFrame. Padrão: 6.
        scoring (tuple, opcional): Métricas de avaliação. Padrão: ('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2').

    Returns:
        DataFrame: DataFrame com os resultados da validação cruzada.

    '''

    resultados = {}  # Dicionário para armazenar os resultados de cada modelo

    for nome_modelo, regressor_config in regressors.items():  # Itera sobre os modelos
        # Constrói o pipeline do modelo
        model = fnc_construir_pipeline_modelo_regressao(
            regressor=regressor_config['regressor'],
            preprocessor=regressor_config.get('preprocessor', None),  # Obtém o pré-processador, se existir
            target_transformer=regressor_config.get('target_transformer', None),  # Obtém o transformador de alvo, se existir
            nome_saida_preprocessor=nome_saida_preprocessor,
            nome_saida_regressor=nome_saida_regressor,
        )

        # Configura a validação cruzada KFold
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        # Realiza a validação cruzada
        scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=kf,
            scoring=scoring,
        )

        resultados[nome_modelo] = scores  # Armazena os resultados do modelo

    return fnc_organiza_resultados(resultados)  # Organiza os resultados em um DataFrame e retorna
