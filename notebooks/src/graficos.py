from IPython.display import display
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from .config import RANDOM_STATE

# ao importar esse arquivo graficos.py, o tema abaixo é aplicado no notebook
contextos = ('paper', 'notebook', 'talk', 'poster')
estilos = ('white', 'dark', 'whitegrid', 'darkgrid', 'ticks')
paletas = ('deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
            'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
            'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c')
sns.set_theme(context=contextos[1], style=estilos[2], palette=paletas[5])

PALETTE_TEMPERATURA = 'coolwarm' # 'coolwarm', 'bwr', 'seismic', 'Reds'
SCATTER_ALPHA = 0.2



#################################################################################
# Função que plota um gráfico de barras horizontais dos coeficientes de um modelo
#################################################################################
def plot_coeficientes(
        df_coefs: pd.DataFrame,
        titulo: str= 'Coeficientes',
        xlabel: str= None,
        ylabel: str= 'Features (Preditores)',
        n_barras: int= 20,
        figsize: list|tuple= (10, 6),
        precisao: int= 3,
        cor_coeficiente: str= 'C0',
        valor_feature_descatada: float= 0.0,
        descricao_feature_descatada: str= 'zero',
        flg_destacar_negativos_com_parenteses: bool= True,
) -> None:
    '''
    Plota um gráfico de barras horizontais dos coeficientes de um modelo.

    Args:
        df_coefs (pd.DataFrame): DataFrame com os coeficientes.
        titulo (str): Título do gráfico.
        xlabel (str, opcional): Rótulo para o eixo x. Se None, o eixo x não terá rótulo.
        ylabel (str, opcional): Rótulo para o eixo y.
        n_barras (int, opcional): Número máximo de barras a serem exibidas.
        figsize (list ou tuple, opcional): Tamanho da figura (largura, altura) em polegadas.
        precisao (int, opcional): Número de casas decimais para os coeficientes.
        cor_coeficiente (str, opcional): Cor das barras e dos rótulos dos coeficientes.
        valor_feature_descatada (float, opcional): Valor absoluto do coeficiente considerado como 'descartado'.
        descricao_feature_descatada (str, opcional): Descrição para features com coeficiente considerado 'descartado.
        flg_destacar_negativos_com_parenteses (bool, opcional): Se True, números negativos terão parênteses.

    Returns:
        None. O gráfico é exibido diretamente.
    '''

    # Seleciona os n coeficientes com maiores valores absolutos
    df_coefs_top = df_coefs.head(n_barras)

    # Cria a figura e os eixos
    fig, ax = plt.subplots(figsize=figsize)

    # Cria o gráfico de barras horizontais com Seaborn, usando o eixo 'ax'
    sns.barplot(x=df_coefs_top.iloc[:, 0], y=df_coefs_top.index, ax=ax, color=cor_coeficiente)

    # Adiciona os números dos coeficientes em frente às barras
    for i, coeficiente in enumerate(df_coefs_top.iloc[:, 0]):
        # Define valor absoluto para comparação
        valor_feature_descatada = abs(valor_feature_descatada)
        # Verifica se o coeficiente está dentro do limiar de descarte
        if valor_feature_descatada * -1 <= coeficiente <= valor_feature_descatada:
            texto_coeficiente = descricao_feature_descatada
        else:
            # Formata o número com a precisão definida
            texto_coeficiente = f'{coeficiente:,.{precisao}f}'.replace('.','¬').replace(',','.').replace('¬',',')
            # Adiciona parênteses para números negativos, se a flag estiver ativa
            if flg_destacar_negativos_com_parenteses and coeficiente < 0.0:
                texto_coeficiente = texto_coeficiente.join(('(', ')'))

        # Adiciona o texto do coeficiente ao gráfico, com ajustes de estilo
        ax.text(
            x=coeficiente if coeficiente > 0.0 else 0.0, # Coeficientes negativos são escritos à direita da barra
            y=i,
            s= texto_coeficiente,
            color=cor_coeficiente,
            va='center',
            fontsize=10, # Ajuste o tamanho da fonte para melhor legibilidade
        )

    # Configurações do gráfico, com ajustes de estilo
    ax.set_title(titulo, weight='bold') # Título em negrito
    ax.axvline(x=0, linestyle='--') # Linha vertical no zero tracejada
    ax.set_xlabel(xlabel, fontsize=12) # Rótulo do eixo x com tamanho maior
    ax.set_ylabel(ylabel, fontsize=12) # Rótulo do eixo y com tamanho maior
    # Remove rótulos do eixo x se xlabel for None
    if xlabel is None:
        ax.set_xticklabels(())

    # Ajusta as margens do gráfico para melhor aproveitamento do espaço
    plt.tight_layout()

    plt.show()





##################################################################################
# Gera um conjunto de gráficos para análise de resíduos de um modelo de regressão.
##################################################################################

# def plot_residuos(
#     y_true: np.ndarray | pd.Series,
#     y_pred: np.ndarray | pd.Series,
#     figsize: tuple = (15, 7),  # Ajustado tamanho da figura
#     hist_bins: int = 30,  # para controlar as bins do histograma
#     hist_color: str = 'C0',  # para controlar a cor do histograma
#     kde_color: str = 'C1',  # para controlar a cor da linha KDE
#     residual_color: str = 'C0',  # para controlar a cor dos pontos no gráfico de resíduos
#     actual_color: str = 'C0',  # para controlar a cor dos pontos no gráfico de valores reais
#     title_fontsize: int = 16,  # Tamanho da fonte para os títulos
#     label_fontsize: int = 14  # Tamanho da fonte para os rótulos dos eixos
# ) -> None: # Gera um conjunto de gráficos
#     '''
#     Gera um conjunto de gráficos para análise de resíduos de um modelo de regressão.

#     Args:
#         y_true (np.ndarray ou pd.Series): Valores reais da variável alvo.
#         y_pred (np.ndarray ou pd.Series): Valores preditos pelo modelo.
#         figsize (tuple, opcional): Tamanho da figura (largura, altura) em polegadas.
#         hist_bins (int, opcional): Número de bins para o histograma de resíduos.
#         hist_color (str, opcional): Cor das barras do histograma.
#         kde_color (str, opcional): Cor da linha KDE no histograma.
#         residual_color (str, opcional): Cor dos pontos no gráfico de resíduos vs preditos.
#         actual_color (str, opcional): Cor dos pontos no gráfico de valores reais vs preditos.
#         title_fontsize (int, opcional): Tamanho da fonte para os títulos dos subplots.
#         label_fontsize (int, opcional): Tamanho da fonte para os rótulos dos eixos.

#     Returns:
#         None. Os gráficos são exibidos diretamente.
#     '''

#     residuos = y_true - y_pred

#     fig, axs = plt.subplots(1, 3, figsize=figsize)

#     # Histograma dos resíduos com KDE
#     sns.histplot(residuos, kde=True, ax=axs[0], bins=hist_bins, color=hist_color, kde_kws={'color': kde_color})
#     axs[0].set_title('Histograma dos Resíduos', fontsize=title_fontsize)
#     axs[0].set_xlabel('Resíduos', fontsize=label_fontsize)
#     axs[0].set_ylabel('Frequência', fontsize=label_fontsize)

#     # Gráfico de resíduos vs valores preditos
#     error_display_01 = PredictionErrorDisplay.from_predictions(
#         y_true=y_true, y_pred=y_pred, kind='residual_vs_predicted', ax=axs[1], scatter_kwargs={'color': residual_color}
#     )
#     axs[1].set_title('Resíduos vs Valores Preditos', fontsize=title_fontsize)
#     axs[1].set_xlabel('Valores Preditos', fontsize=label_fontsize)
#     axs[1].set_ylabel('Resíduos', fontsize=label_fontsize)

#     # Gráfico de valores reais vs valores preditos
#     error_display_02 = PredictionErrorDisplay.from_predictions(
#         y_true=y_true, y_pred=y_pred, kind='actual_vs_predicted', ax=axs[2], scatter_kwargs={'color': actual_color}
#     )
#     axs[2].set_title('Valores Reais vs Valores Preditos', fontsize=title_fontsize)
#     axs[2].set_xlabel('Valores Preditos', fontsize=label_fontsize)
#     axs[2].set_ylabel('Valores Reais', fontsize=label_fontsize)


#     plt.tight_layout()
#     plt.show()








################################################################################
# Função que plota gráficos de resíduos para um estimador
################################################################################
def plot_residuos_estimador(
    estimator,
    X: list | dict,
    y: list | dict,
    eng_formatter: bool = True,
    fracao_amostra: float = 0.2,
    figsize: list|tuple = (10, 12),
    mosaico: str = 'AA;BC',
    espacamento: float = 0.4,
    random_state: int = RANDOM_STATE,
    titulo: str = 'Gráficos de Resíduos',
    cor_residuos: str = 'C0',
    cor_histograma: str = 'C0',
    kde: bool= True,
    bins: int= 30,
    alpha: float = 0.5,
) -> None:
    '''
        Plota gráficos de resíduos para um estimador, incluindo:
        - Histograma dos resíduos.
        - Gráfico de resíduos versus valores previstos.
        - Gráfico de valores reais versus valores previstos.

        Parâmetros:
        ----------
        estimator : Any
            O modelo de estimador que será usado para prever os valores.
        X : Union[list, Dict[str, Any]]
            Conjunto de dados de entrada para fazer previsões.
        y : Union[list, Dict[str, Any]]
            Valores reais a serem comparados com as previsões.
        eng_formatter : bool, opcional
            Se True, aplica formatação em engenheiro aos eixos.
        fracao_amostra : float, opcional
            Fração da amostra a ser utilizada para o gráfico (0 < fracao_amostra < 1).
        figsize : tuple[int, int], opcional
            Tamanho da figura (largura, altura) em polegadas.
        mosaico : str, opcional
            Layout dos subgráficos.
        espacamento : float, opcional
            Espaçamento entre os gráficos.
        random_state : int, opcional
            Semente para replicar aleatoriedade.
        titulo : str, opcional
            Título geral dos gráficos.
        cor_residuos : str, opcional
            Cor dos pontos no gráfico de resíduos versus valores previstos.
        cor_histograma : str, opcional
            Cor do histograma dos resíduos.
        kde : bool, opcional
            True para plotar KDE no histograma de resíduos.
        bins: int, opcional
            Número de bins do histograma de resíduos.
        alpha : float, opcional
            Transparência dos pontos nos gráficos.

        Retorna:
        --------
        None
            A função não retorna nenhum valor; apenas exibe os gráficos.
    '''
    
    # Define o espaçamento entre os gráficos
    ESPACAMENTO = {'wspace': espacamento, 'hspace': espacamento}

    # Cria uma nova figura com o tamanho especificado
    fig = plt.figure(figsize=figsize)
    
    # Cria uma grade de subgráficos de acordo com o mosaico especificado
    axs = fig.subplot_mosaic(mosaic=mosaico, gridspec_kw=ESPACAMENTO)

    # Pega os itens do mosaico ordenada
    itens_mosaico = set(''.join(mosaico))
    itens_mosaico.discard(';')
    itens_mosaico = sorted(itens_mosaico)

    # Define o título da figura
    fig.suptitle(titulo, fontsize=16)


    # Cria o gráfico de resíduos versus valores previstos
    ax = axs[itens_mosaico[1]]
    residual_vs_predicted = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind='residual_vs_predicted',
        ax=axs[itens_mosaico[1]],
        random_state=random_state,
        scatter_kwargs={'alpha': alpha, 'color': cor_residuos},
        subsample=fracao_amostra,
    )
    ax.set_xlabel('Predito')
    ax.set_ylabel('Resíduos (real - predito)')


    # Cria o gráfico de valores reais versus valores previstos
    ax = axs[itens_mosaico[2]]
    actual_vs_predicted = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind='actual_vs_predicted',
        ax=axs[itens_mosaico[2]],
        random_state=random_state,
        scatter_kwargs={'alpha': alpha, 'color': cor_residuos},
        subsample=fracao_amostra,
    )
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')


    # Calcula os resíduos
    residuos = residual_vs_predicted.y_true - residual_vs_predicted.y_pred

    # Cria um histograma dos resíduos
    ax = axs[itens_mosaico[0]]
    sns.histplot(residuos, ax=ax, color=cor_histograma, kde=kde, bins=bins, legend=False)
    ax.set_xlabel('Resíduos (real - predito)')
    ax.set_ylabel('Quantidade')


    # Aplica formatação em engenheiro se especificado
    for ax in itens_mosaico:
        axs[ax].tick_params(
            axis='x',
            labelrotation=50,
        )
        if eng_formatter:
            axs[ax].yaxis.set_major_formatter(EngFormatter(places=0))
            axs[ax].xaxis.set_major_formatter(EngFormatter(places=0))



    # Exibe a figura
    plt.show()








########################################################################################################
# função para plotar boxplots de diferentes modelos para diferentes méticas de um dataframe de resultado
########################################################################################################
def plot_comparar_metricas_modelos(
    df_resultados: pd.DataFrame,  # DataFrame contendo os resultados dos modelos
    comparar_metricas: list|tuple = (),  # Lista/tupla de métricas a serem comparadas
    nomes_metricas: list|tuple = (),  # Nomes para as métricas nos gráficos
    figsize: list|tuple = (10, 8),  # Tamanho da figura (largura, altura)
    colunas_graficos: int = 1,  # Número de colunas de gráficos
    flg_boxplots_horizontais: bool = False,  # Se True, plota boxplots horizontalmente
    cor_boxplot: str = None,  # Cor dos boxplots
    titulo_grafico: str = 'Comparação de Métricas entre Modelos',  # Título geral do gráfico
    tamanho_titulo: int = 16,  # Tamanho da fonte do título
    tamanho_label: int = 14  # Tamanho da fonte dos labels dos eixos
) -> None:
    '''
    Gera boxplots comparando diferentes modelos em diversas métricas.

    Args:
        df_resultados (pd.DataFrame): DataFrame contendo os resultados dos modelos.
        comparar_metricas (list|tuple, opcional): Lista/tupla de métricas a serem comparadas. 
            Se vazio, compara todas as colunas numéricas.
        nomes_metricas (list|tuple, opcional): Nomes para as métricas nos gráficos. 
            Se não fornecidos, usa os nomes das colunas.
        figsize (list|tuple, opcional): Tamanho da figura (largura, altura) em polegadas.
        colunas_graficos (int, opcional): Número de colunas de gráficos.
        flg_boxplots_horizontais (bool, opcional): Se True, plota boxplots horizontalmente.
        cor_boxplot (str, opcional): Cor dos boxplots. Se None, usa a cor padrão do Seaborn.
        titulo_grafico (str, opcional): Título geral do gráfico.
        tamanho_titulo (int, opcional): Tamanho da fonte do título.
        tamanho_label (int, opcional): Tamanho da fonte dos labels dos eixos.

    Returns:
        None. Exibe os gráficos.

    Raises:
        TypeError: Se `df_resultados` não for um DataFrame.
    '''

    # Verifica se df_resultados é um DataFrame
    if not isinstance(df_resultados, pd.DataFrame):
        raise TypeError('df_resultados deve ser um DataFrame.')

    # Obtém as métricas a serem comparadas
    quantidade_colunas = len(comparar_metricas)

    # Se a lista de métricas estiver vazia, usa todas as colunas numéricas
    if quantidade_colunas == 0:
        comparar_metricas = df_resultados.select_dtypes(include='number').columns.to_list()
        quantidade_colunas = len(comparar_metricas)
        # Se não houver colunas numéricas, retorna None
        if quantidade_colunas == 0:
            return None

    # Se os nomes das métricas não forem fornecidos, usa os nomes das colunas
    if len(nomes_metricas) != quantidade_colunas:
        nomes_metricas = comparar_metricas

    # Define o número de colunas de gráficos
    if colunas_graficos <= 0:
        colunas_graficos = 1

    # Calcula o número de linhas de gráficos
    linhas_graficos = ceil(quantidade_colunas / colunas_graficos)

    # Define o compartilhamento dos eixos
    if flg_boxplots_horizontais:
        sharex = False
        sharey = True
    else:
        sharex = True
        sharey = False

    # Cria a figura e os subplots
    fig, axs = plt.subplots(nrows=linhas_graficos, ncols=colunas_graficos, figsize=figsize, sharex=sharex, sharey=sharey)
    # Adiciona o título geral do gráfico
    fig.suptitle(titulo_grafico, fontsize=tamanho_titulo)

    # Itera sobre os subplots, métricas e nomes de métricas
    for ax, metrica, nome_metrica in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        # Define os eixos x e y de acordo com a orientação dos boxplots
        if flg_boxplots_horizontais:
            x = metrica
            y = 'model'
        else:
            x = 'model'
            y = metrica

        # Cria o boxplot
        sns.boxplot(
            data=df_resultados,
            x=x,
            y=y,
            showmeans=True,  # Mostra a média
            ax=ax,
            color=cor_boxplot  # Define a cor do boxplot
        )

        # Configura os labels dos eixos
        if flg_boxplots_horizontais:
            ax.set_xlabel(nome_metrica, fontsize=tamanho_label)
            ax.set_ylabel(None)
        else:
            ax.set_ylabel(nome_metrica, fontsize=tamanho_label)
            ax.set_xlabel(None)

        # Configura o título do subplot
        ax.set_title(nome_metrica, weight='bold', fontsize=tamanho_label)
        # Rotaciona os ticks do eixo x
        ax.tick_params(axis='x', rotation=90, labelsize=tamanho_label)

    # Ajusta o layout para evitar sobreposição, considerando o título geral
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()






###################################################################################
# Função para mostrar e/ou retornar as correlações entre as colunas de um dataframe
###################################################################################
def fnc_correlacao_formatada(
    df: pd.DataFrame,  # DataFrame com os dados (apenas colunas numéricas serão consideradas)
    flg_mostrar_heatmap: bool = True,  # Se True, mostra o heatmap
    flg_mostrar_heatmap_triangular_superior: bool = False,  # Se True, mostra também a triangular superior do heatmap
    flg_heatmap_primeiro: bool = True,  # Se True, mostra o heatmap antes do dataframe
    flg_mostrar_dataframe: bool = False,  # Se True, mostra o dataframe com as correlações
    flg_retornar_dataframe: bool = False,  # Se True, retorna o dataframe com as correlações
    coluna_target: str = None,  # Se None, mostra/retorna todas as correlações. Se informado a coluna target, mostra/retorna apenas as correlações com a coluna target no dataframe
    palette: str = PALETTE_TEMPERATURA,  # Paleta de cores para o heatmap ('coolwarm', 'bwr', 'seismic', 'Reds', 'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r')
    figsize: tuple = (10, 10),  # Tamanho da figura
    precisao: int = 2,  # Número de casas decimais
) -> pd.DataFrame | None:  # Retorna None se flg_retornar_dataframe for False. Retorna o dataframe com as correlações se flg_retornar_dataframe for True
    '''
    Exibe e/ou retorna a matriz de correlação de um DataFrame, opcionalmente incluindo um heatmap.

    Esta função calcula a matriz de correlação de um DataFrame, permitindo a visualização
    através de um heatmap e/ou a exibição/retorno da matriz em formato de DataFrame.
    É possível especificar uma coluna alvo para calcular correlações apenas com ela.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados numéricos.
        flg_mostrar_heatmap (bool, opcional): Se True, exibe o heatmap da matriz de correlação. Padrão: True.
        flg_mostrar_heatmap_triangular_superior (bool, opcional): Se True, exibe também a triangular superior do heatmap.
            Útil quando `flg_mostrar_heatmap` é True. Padrão: False.
        flg_heatmap_primeiro (bool, opcional): Se True, exibe o heatmap antes do DataFrame. Padrão: True.
        flg_mostrar_dataframe (bool, opcional): Se True, exibe a matriz de correlação como um DataFrame. Padrão: False.
        flg_retornar_dataframe (bool, opcional): Se True, retorna a matriz de correlação como um DataFrame. Padrão: False.
        coluna_target (str, opcional): Nome da coluna para calcular correlações apenas com ela. Se None, calcula
            correlações entre todas as colunas. Padrão: None.
        palette (str, opcional): Paleta de cores para o heatmap. Opções: 'coolwarm', 'bwr', 'seismic', 'Reds', 
            'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r'. Padrão: 'coolwarm'.
        figsize (tuple, opcional): Tamanho da figura do heatmap (largura, altura) em polegadas. Padrão: (10, 10).
        precisao (int, opcional): Número de casas decimais para exibir as correlações no DataFrame. Padrão: 2.

    Returns:
        pd.DataFrame | None: Se `flg_retornar_dataframe` for True, retorna a matriz de correlação como um DataFrame.
            Caso contrário, retorna None.

    Raises:
        TypeError: Se `df` não for um DataFrame.

    Examples:
        Para exibir o heatmap e o DataFrame de correlação:
        >>> fnc_correlacao_formatada(df, flg_mostrar_heatmap=True, flg_mostrar_dataframe=True)

        Para retornar apenas o DataFrame de correlação:
        >>> matriz_correlacao = fnc_correlacao_formatada(df, flg_retornar_dataframe=True)

        Para calcular correlações apenas com a coluna 'target':
        >>> fnc_correlacao_formatada(df, coluna_target='target')
    '''

    # Verifica se pelo menos uma das flags de visualização ou retorno está ativa
    if flg_mostrar_heatmap or flg_mostrar_dataframe or flg_retornar_dataframe:
        # Seleciona apenas colunas numéricas e calcula a matriz de correlação
        df_corr = df.select_dtypes(include='number').corr()
        # Se o DataFrame estiver vazio após a seleção de colunas numéricas, retorna None
        if df_corr.empty:
            return None
    else:
        return None  # Se nenhuma opção de visualização ou retorno estiver habilitada, retorna None

    # Função interna para mostrar o heatmap
    def subfnc_mostrar_heatmap():
        if flg_mostrar_heatmap:
            # Cria a figura e o eixo
            fig, ax = plt.subplots(figsize=figsize)

            # Cria o heatmap com seaborn
            sns.heatmap(
                data=df_corr,
                mask=None if flg_mostrar_heatmap_triangular_superior else np.triu(df_corr),  # Mascara a parte superior do triângulo se necessário
                annot=True,  # Exibe os valores das correlações nas células
                fmt=f'.{precisao}f',  # Formata os valores com a precisão especificada
                ax=ax,  # Usa o eixo criado
                cmap=palette  # Usa a paleta de cores especificada
            )

            plt.show()  # Exibe o gráfico


    # Função interna para mostrar o DataFrame
    def subfnc_mostrar_dataframe():
        if flg_mostrar_dataframe:
            # Exibe o DataFrame com estilo condicional usando display
            if coluna_target is None:
                display(df_corr.style.background_gradient(cmap=palette, vmin=-1, vmax=1).format(precision=precisao))  # Estilo para todas as colunas
            else:
                display(df_corr[[coluna_target]].sort_values(by=[coluna_target]).T.style.background_gradient(cmap=palette, vmin=-1, vmax=1).format(precision=precisao))  # Estilo para coluna target


    # Controla a ordem de exibição do heatmap e do DataFrame
    if flg_heatmap_primeiro:
        subfnc_mostrar_heatmap()  # Exibe o heatmap primeiro
        subfnc_mostrar_dataframe()  # Exibe o DataFrame depois
    else:
        subfnc_mostrar_dataframe()  # Exibe o DataFrame primeiro
        subfnc_mostrar_heatmap()  # Exibe o heatmap depois


    # Retorna o DataFrame se a flag estiver ativa
    if flg_retornar_dataframe:
        if coluna_target is None:
            return df_corr  # Retorna o DataFrame completo
        else:
            return df_corr[[coluna_target]]  # Retorna apenas a coluna target
    else:
        return None  # Retorna None se a flag não estiver ativa
################################################################################
