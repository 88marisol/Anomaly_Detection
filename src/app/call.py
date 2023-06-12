import requests

# URL de la API Gateway
api_url = 'https://w8063fiule.execute-api.us-east-1.amazonaws.com/test'

# Datos de entrada para la solicitud
data = {
  "data1": [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.001243831,
    0.001243831,
    0.003731494,
    0.001243831,
    3.941372454,
    0.019209278,
    0.006403093,
    0.000674601,
    0.013119502,
    0.004948772,
    0.013858023,
    0.014886165,
    0.012642225,
    0.033216963,
    0.017751479,
    0.012642225,
    0.026760563,
    0.012642225,
    0.022222222,
    0.018359853,
    0.022222222,
    0.012642225,
    0.022222222,
    0.012642225,
    0.013142376,
    0.683309243,
    0.712245884,
    0.691978626,
    0.909751716,
    0.755820215,
    0.691978626,
    0.691978626,
    0.846484216,
    0.691978626,
    0.743338278,
    0.707699017,
    0.743338278,
    0.691978626,
    0.743338278,
    0.678732125,
    0.248857394,
    1.91964055,
    1,
    0.003543125,
    2.267889681,
    1,
    0,
    0,
    0.248857394,
    1.91964055,
    0.496310118,
    1.052606042,
    0.011971152,
    0.024875257,
    0.204180469,
    0.248857394,
    0.003543125,
    0.024875257,
    0.024875257,
    0.066983941,
    0.001886872,
    0.024875257,
    0.847346023,
    0.011646191,
    1.476614531,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    0.006586312,
    0.6,
    0.01487696,
    0.814720118,
    0.118771909,
    0.457900278,
    0.304553303,
    0.470491278,
    0.354164469,
    0.383370441,
    0.15917326,
    0.429797644,
    0.14480892,
    0.645235405,
    0.300055921,
    0.455223709,
    0.314696213,
    0.43748886,
    0.300055921,
    0.455223709,
    0.316825415,
    0.418016162,
    0.216730774,
    0.507736906,
    0.316825415,
    0.418016162,
    0.304553303,
    0.470491278,
    0.315057312,
    0.427795638,
    0.018343597,
    0.365359798,
    0,
    0.13540406
  ],
  "data": [1,1,0,0,0,0,0,0,0,0,0,0,0,0.208169291,0.208169291,0.208169291,0.208169291,4.479924457,0,0,0,0,0,0.016393443,0.024927879,0.016393443,0.116695759,0.011019284,0.016393443,0.016393443,0.016393443,0.011019284,0.101565719,0.011019284,0.016393443,0.011019284,0.016913319,0.037037037,0.652575512,0.679282254,0.652575512,0.909751716,0.649230287,0.653316886,0.652575512,0.652575512,0.652575512,0.649230287,0.867539993,0.649230287,0.652575512,0.649230287,0.629648762,0.138319672,1.873167604,0,0.37192623,2.483246292,0,0,1,0.138319672,1.873167604,1,0,110.8114754,1,19.95696721,0.138319672,0.37192623,0.969262295,1,1,0.37192623,0.969262295,0.198078027,0.433401639,1.854778603,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0.008511054,0.2,0.008334158,0.842027973,0.119910692,0.47008395,0.293118398,0.492447767,0.355416141,0.381393017,0.138722326,0.47787312,0.131034323,0.674486143,0.289007276,0.476151621,0.266802021,0.544571107,0.289007276,0.476151621,0.291394416,0.464583114,0.253399975,0.429362725,0.291394416,0.464583114,0.293118398,0.492447767,0.290816862,0.474567528,0.016276343,0.383475311,0.676864982,0.989039522]
}

# Realizar la solicitud POST a la API Gateway
response = requests.post(api_url, json=data, verify=False)

# Obtener la respuesta de la API
if response.status_code == 200:
    # La solicitud fue exitosa
    response_data = response.json()
    print('Respuesta:', response_data)
else:
    # La solicitud no fue exitosa
    print('Error en la solicitud. Código de estado:', response.status_code)
