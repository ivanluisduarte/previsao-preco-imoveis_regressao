# %% IMPORTAÇÕES

# comentado para otimização, só é carregado na chamada inicial importado na função com cache
# from joblib import load # para importar o modelo e os dados geográficos

from pandas import cut, read_parquet
from pydeck import Deck, Layer, ViewState
import streamlit as st # interface WEB - https://streamlit.io/

# comentado para otimização, só é carregado na chamada inicial importado na função com cache
# arquivos utilizados
# from notebooks.src.config import(
#     DADOS_GEO_DATAFRAME,
#     DADOS_MEDIAN,
#     MODELO_FINAL,
# )


################################################################################
# %% FUNÇÕES CACHE_DATA
# qualquer coisa que possa ser armazenado em database
# Python primitives, dataframe e API calls

@st.cache_data
def carregar_dados_geograficos():
    from joblib import load
    from notebooks.src.config import DADOS_GEO_DATAFRAME

    return load(DADOS_GEO_DATAFRAME)


@st.cache_data
def carregar_dados_medianas():
    from notebooks.src.config import DADOS_MEDIAN

    return read_parquet(DADOS_MEDIAN)


@st.cache_data
def carregar_categorias_ordenadas():
    from joblib import load
    from notebooks.src.config import DCT_CATEGORICAS_ORDENADAS

    return load(DCT_CATEGORICAS_ORDENADAS)


################################################################################
# %% FUNÇÕES CACHE_RESOURCE
# qualquer coisa que NÃO possa ser armazenado em database
# ML models e database connections

@st.cache_resource
def carregar_modelo():
    from joblib import load
    from notebooks.src.config import MODELO_FINAL

    return load(MODELO_FINAL)

################################################################################
# %% carregando arquivos ou seu cache, se já existir

gdf_geo = carregar_dados_geograficos()
df_medianas = carregar_dados_medianas()
dct_categoricas_ordenadas = carregar_categorias_ordenadas()
modelo = carregar_modelo()

################################################################################
# %% PAGINA WEB
titulo = 'Previsão de preços de imóveis na Califórnia'
st.set_page_config(
    page_title=titulo,
    page_icon=':house:', # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
    layout='wide',
)
st.title(
    body=titulo,
    help='''Não utilize para fins reais.
Foram utilizados dados de 1990 e não refletem a realidade dos dias de hoje.
O objetivo deste trabalho é apenas para compor o portfólio de ciência de dados de Ivan Luís Duarte.
https://github.com/ivanluisduarte/previsao-preco-imoveis_regressao

GitHub: https://github.com/ivanluisduarte

LinkedIn: https://www.linkedin.com/in/ivanluisduarte/
''',
)

# dividindo a tela em 2 colunas - formulário e mapa
coluna1, coluna2 = st.columns(
    spec=(0.3, 0.7),
    gap='small',
    border=True
)


with coluna1:
    # inputs - formulário

    with st.form(
        key='formulario',
        clear_on_submit=False,
        border=True,
    ):
        selecionar_condados = st.selectbox(
            label='Condado',
            options=df_medianas.index,
            help='Condados o estado da Califórnia-EUA'
        )
        housing_median_age = st.number_input(
            label='Idade do imóvel',
            min_value=1,
            max_value=50,
            value=10,
            format='%d',
            help='Tempo de existência do imóvel em anos'
        )
        median_income = st.slider(
            label='Renda média anual (milhares de US$)',
            min_value=5,
            max_value=100,
            value=45,
            step=5,
            format='%d',
            help='Renda média anual em dólares para a população da região'
        )

        st.form_submit_button(
            label='Prever preço',
            help='''Ao clicar nesse botão, uma previsão será feita para o valor aproximado
do imóvel e se selecionado um mapa pode ser apresentado com destaque para o Condado.'''
        )


################################################################################
# %% PASSANDO OS DADOS PELO MODELO

    df_valores_condado = df_medianas.loc[selecionar_condados].to_frame().T

    # os valores estão multiplicados por 10 mil, e estamos convertendo para 1 mil
    # na apresentação, no modelo temos que ter o valor na escala correta
    df_valores_condado['median_income'] = median_income / 10

    df_valores_condado['median_income_cat'] = cut(
        x=df_valores_condado['median_income'],
        bins=dct_categoricas_ordenadas['bins_median_income_cat'],
        labels=dct_categoricas_ordenadas['labels_median_income_cat'],
        right=False,
        ordered=True,
    )

    df_valores_condado['housing_median_age'] = housing_median_age

    df_valores_condado['housing_median_age_cat'] = cut(
        x=df_valores_condado['housing_median_age'],
        bins=dct_categoricas_ordenadas['bins_housing_median_age_cat'],
        labels=dct_categoricas_ordenadas['labels_housing_median_age_cat'],
        right=False,
        ordered=True,
    )

    df_valores_condado['population_cat'] = cut(
        x=df_valores_condado['population'],
        bins=dct_categoricas_ordenadas['bins_population_cat'],
        labels=dct_categoricas_ordenadas['labels_population_cat'],
        right=False,
        ordered=True,
    )

    # aux = df_valores_condado.to_frame().T
    # faz a predição com os dados da tela
    preco = modelo.predict(df_valores_condado)

    st.metric( # mostrando o preço
        label='Preço previsto (US$)',
        value= f'{preco[0][0]:,.2f}'.replace('.', '¬').replace(',', '.').replace('¬', ','),
        help='''Esse valor é apenas uma estimativa, baseado nos dados fornecidos e nas medianas
de diversos indicadores do condado.Faça uma análise mais profunda do seu imóvel em expecífico.'''
    )


################################################################################
# %% CONSTRUINDO O MAPA

with coluna2:

    # localização inicial - o condado selecionado fica no centro da visualização do mapa
    visualizacao_inicial = ViewState(
        latitude = float(df_valores_condado['latitude']),
        longitude = float(df_valores_condado['longitude']),
        zoom=4,
        min_zoom=3,
        max_zoom=10,
    )

    # hint apresentado com o nome do condado quando colocamos o mouse em cima
    tooltip = {
        'html': '<b>Condado:</b> {condado}',
        'style': {
            'backgroundcolor': 'steelblue',
            'color': 'white',
            'fontsize': '10px',
        },
    }

    # colore o estado da califórnia
    camada_california = Layer(
        type='GeoJsonLayer',
        data=gdf_geo[['condado', 'geometry']],
        get_polygon='geometry',
        get_fill_color=[0, 0, 255, 55], # RGB + alfa
        get_line_color=[255, 255, 255], # branco
        get_line_width=1000,
        pickable=True, # necessário para funcionar o tooltip
        auto_highlight = True,
    )

    # colore de cor diferente o condado selecionado
    camada_condado_selecionado = Layer(
        type='GeoJsonLayer',
        data=gdf_geo[gdf_geo['condado'] == selecionar_condados][['condado', 'geometry']],
        get_polygon='geometry',
        get_fill_color=[255, 0, 0, 55], # RGB + alfa
        get_line_color=[0, 0, 0], # preto
        get_line_width=1000,
        pickable=True, # necessário para funcionar o tooltip
        auto_highlight = True,
    )

    # mapa
    mapa = Deck(
        initial_view_state=visualizacao_inicial,
        map_style='dark',
        layers=[
            camada_california,
            camada_condado_selecionado,
        ],
        tooltip=tooltip,
    )

    # plotando o mapa
    st.pydeck_chart(
        pydeck_obj=mapa
    )