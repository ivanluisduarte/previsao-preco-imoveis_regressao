from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / 'dados'

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_ORIGINAIS = PASTA_DADOS / 'archive.zip' # download de https://www.kaggle.com/datasets/camnugent/california-housing-prices/data
DADOS_COMPLETOS = PASTA_DADOS / 'housing_complete.parquet'
DADOS_LIMPOS = PASTA_DADOS / 'housing_clean.parquet'

DADOS_X_TRAIN = PASTA_DADOS / 'x_train.parquet'
DADOS_X_TEST = PASTA_DADOS / 'x_test.parquet'
DADOS_X_VALIDATION = PASTA_DADOS / 'x_validation.parquet'

DADOS_Y_TRAIN = PASTA_DADOS / 'y_train.parquet'
DADOS_Y_TEST = PASTA_DADOS / 'y_test.parquet'
DADOS_Y_VALIDATION = PASTA_DADOS / 'y_validation.parquet'

DADOS_GEO_DATAFRAME = PASTA_DADOS / 'california-counties.joblib'
DADOS_GEO_ORIGINAIS = PASTA_DADOS / 'california-counties.geojson' # download de https://github.com/codeforgermany/click_that_hood/blob/main/public/data/california-counties.geojson

DADOS_MEDIAN = PASTA_DADOS / 'df_counties_median.parquet'

# # coloque abaixo o caminho para os arquivos de modelos de seu projeto
PASTA_MODELOS = PASTA_PROJETO / 'modelos'
DCT_CATEGORICAS_ORDENADAS = PASTA_MODELOS / 'dct_categoricas_ordenadas.joblib'
MODELO_FINAL = PASTA_MODELOS / 'lr_polyfeat_target_quantile.joblib'


RANDOM_STATE = 255 # semente para reprodução dos resultados

