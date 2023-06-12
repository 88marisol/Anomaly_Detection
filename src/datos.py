import os
import json
import pandas as pd
import re
import ast
from collections import Counter
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from scipy.stats import entropy
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
from bs4 import BeautifulSoup
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Obtener la ruta del archivo JSON en la carpeta actual
json_file = os.getcwd()+"\\config\\config.json"
# Leer el archivo JSON
with open(json_file, 'r') as f:
    cfg = json.load(f)

# Carga el conjunto de datos en un DataFrame de pandas
data = pd.read_csv(cfg['root_folder'] + "\\data\\"+cfg['data_name'])
data['timestamp'] = data['timestamp'].str.replace('.', '', regex=False).astype(float)
#data['timestamp_date'] = pd.to_datetime(data['timestamp'])
data['userId'] = data['userId.1']
data = data.drop('userId.1', axis=1)


def extract_device(user_agent):
    if 'iPhone' in user_agent:
        return 'iPhone'
    elif 'Macintosh' in user_agent:
        return 'Macintosh'
    elif 'Linux' in user_agent:
        return 'Linux'
    elif 'Android' in user_agent:
        return 'Android'
    elif 'Windows' in user_agent:
        return 'Windows'
    else:
        return 'Other'

# Calcular la probabilidad por usuario
def calc_stat(data,us,variable):
    prob_df = data.groupby([us, variable]).size().div(data.groupby(us).size(), level=us).reset_index()
    prob_df.columns = [us, variable, 'probability']

    # Calcular la entropía por usuario
    entropy_df = prob_df.groupby(us)['probability'].apply(lambda x: entropy(x,base=2)).reset_index()
    entropy_df.columns = [us, 'entropy']

    # Combinar los resultados en un solo DataFrame
    return pd.merge(prob_df, entropy_df, on=us)

user_ids = []
args = []
names = []
types = []
values = []
ids = []
process = []
events = []
timestamps = []
for index, row in data.iterrows():
    user_id = row['userId']
    arg = ast.literal_eval(re.sub(r",\s*(?=\])", "", row['args']))
    n = int(row['argsNum'])
    timestamp = row['timestamp']
    pro = row['processName']
    event = row['eventName']
    
    # Iterar sobre los elementos de la lista de 'args'
    for i in range(0,n):
        d = arg[i]
               
        # Agregar los valores a las listas
        user_ids.append(user_id)
        args.append(arg)
        names.append(d['name'])
        types.append(d['type'])
        values.append(d['value'])
        ids.append(id)
        process.append(pro)
        events.append(event)
        timestamps.append(timestamp)

# Crear el nuevo DataFrame
new_data = pd.DataFrame({'id': ids,
                         'args': args,
                         'userid': user_ids,
                         'name': names,
                         'type': types,
                         'value': values,
                         'processName': process,
                         'eventName': events,
                         'timestamp':timestamps
                         })

#tokenizacion
def extract_keywords(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    tokens = word_tokenize(text)
    keywords = [token.lower() for token in tokens if token.isalpha()]
    return keywords

def extract_keywords_dataframe(df):
    df['values'] = df['value'].apply(extract_keywords)
    return df

# extrae palabras clave
df_with_keywords = extract_keywords_dataframe(new_data)

# las palabras más usadas en la columna 'name' para cada usuario
name_most_common = df_with_keywords.groupby('userid')['name'].apply(lambda x: Counter(x).most_common())

# top 'name1', 'name2', 'name3' y 'name4'
for i in range(5):
    df_with_keywords[f'name{i+1}'] = df_with_keywords.apply(lambda row: 1 if row['name'] == name_most_common[row['userid']][i][0] else 0, axis=1)
# palabras más usadas en la columna 'values' para cada usuario
values_common = df_with_keywords.groupby('userid')['values'].apply(lambda x: Counter(value for sublist in x for value in sublist).most_common())

# top 'value1', 'value2', 'value3' y 'value4'
for i in range(6):
    df_with_keywords[f'values{i+1}'] = df_with_keywords.apply(lambda row: 1 if len(row['values']) > i and values_common[row['userid']][i][0] in row['values'] else 0, axis=1)

def check_words_in_text(values, text):
    regex = re.compile(r'\b(?:' + '|'.join(values) + r')\b', flags=re.IGNORECASE)
    return 1 if regex.search(text) is not None else 0

# si 'values' está contenida en 'eventName'
df_with_keywords['values_in_eventName'] = df_with_keywords.apply(lambda row: check_words_in_text(row['values'], row['eventName']) if row['values'] else 0, axis=1)

## si 'values' está contenida en 'processName'
df_with_keywords['values_in_processName'] = df_with_keywords.apply(lambda row: check_words_in_text(row['values'], row['processName']) if row['values'] else 0, axis=1)

## si 'type' es el tipo más comun para el usuario y el name
type_common = df_with_keywords.groupby(['userid', 'name', 'type']).size().reset_index(name='count')
most_used_types = type_common.groupby(['userid', 'name'])['type'].apply(lambda x: x.iloc[0]).reset_index(name='most_used_type')
df_with_keywords = pd.merge(df_with_keywords, most_used_types, on=['userid', 'name'], how='left')
df_with_keywords['type1'] = df_with_keywords.apply(lambda row: 0 if row['type'] == row['most_used_type'] else 1, axis=1)
df_with_keywords['args'] = df_with_keywords['args'].astype(str)

stats_use_namearg = calc_stat(df_with_keywords,'userid','name')

# Función para calcular la probabilidad de una palabra en una lista
def calculate_probability(word_list):
    word_counts = Counter(word_list)
    total_count = len(word_list)
    probabilities = {word: count / total_count for word, count in word_counts.items()}
    return probabilities

# Calcular la probabilidad máxima, mínima y entropía por userid
result = []
for userid, group in df_with_keywords.groupby('userid'):
    word_list = [word for sublist in group['values'] for word in sublist]
    probabilities = calculate_probability(word_list)
    entropy_value = entropy(list(probabilities.values()), base=2)
    result.append({'userid': userid,
                   'probabilities':probabilities,
                   'entropy': entropy_value})

# Crear dataframe con los resultados
result_df = pd.DataFrame(result)

def calculate_probabilities(row):
    values = list(set(row['values']))
    probabilities = row['probabilities_user_valuearg']
    probabilities_matching_values = [probabilities[word] for word in values if word in probabilities]
    total_probability = sum(probabilities_matching_values) if probabilities_matching_values else 0
    min_probability = min(probabilities_matching_values) if probabilities_matching_values else 0
    max_probability = max(probabilities_matching_values) if probabilities_matching_values else 0
    return total_probability, min_probability, max_probability

result_df.columns=['userid', 'probabilities_user_valuearg', 'entropy_user_valuearg']
df_with_keywords = pd.merge(df_with_keywords,result_df,on=['userid'], how='left')
df_with_keywords[['total_probability_user_valuearg', 'min_probability_user_valuearg', 'max_probability_user_valuearg']] = df_with_keywords.apply(calculate_probabilities, axis=1, result_type='expand')
stats_use_namearg.columns=['userid', 'name', 'probability_user_namearg', 'entropy_user_namearg']
# Aplicar la función por fila del dataframe
df_with_keywords = pd.merge(df_with_keywords,stats_use_namearg,on=['userid','name'], how='left')

sumas_words = df_with_keywords.groupby(['timestamp']).agg({'name1': 'sum', 'name2': 'sum','name3': 'sum', 'name4': 'sum','name5': 'sum','values1': 'sum', 'values2': 'sum','values3': 'sum', 'values4': 'sum','values6': 'sum','values_in_processName': 'sum','values_in_eventName': 'sum','type1': 'sum','probability_user_namearg':['min', 'max','sum','mean'],'entropy_user_valuearg':'max','total_probability_user_valuearg':['sum','mean'],'min_probability_user_valuearg':['min','mean'],'max_probability_user_valuearg':['max','mean']}).reset_index()
sumas_words.columns = ['timestamp', 'name1', 'name2', 'name3', 'name4', 'name5', 
                       'values1', 'values2', 'values3', 'values4', 'values5',
                       'values_in_processName', 'values_in_eventName', 'type1',
                       'min_probability_user_namearg', 'max_probability_user_namearg', 'sum_probability_user_namearg',
                        'mean_probability_user_namearg','entropy_user_valuearg','total_probability_user_valuearg', 'mean_probability_user_valuearg',
                        'min_probability_user_valuearg','mean_min_probability_user_valuearg','max_probability_user_valuearg','mean_max_probability_user_valuearg']

data = pd.merge(data, sumas_words, on=['timestamp'], how='left')


# estadisticos por identificadores
stats_user = data.groupby('userId')['timestamp'].agg(['mean', 'std','count'])
stats_user['cv'] = stats_user['std'] / stats_user['mean']
data['perc_user'] = data.groupby('userId')['timestamp'].rank(pct=True)

stats_event = data.groupby('eventId')['timestamp'].agg(['mean', 'std','count'])
stats_event['cv'] = stats_event['std'] / stats_event['mean']
data['perc_event'] = data.groupby('eventId')['timestamp'].rank(pct=True)

stats_process = data.groupby('processId')['timestamp'].agg(['mean', 'std','count'])
stats_process['cv'] = stats_process['std'] / stats_process['mean']
data['perc_process'] = data.groupby('processId')['timestamp'].rank(pct=True)

stats_pprocess = data.groupby('parentProcessId')['timestamp'].agg(['mean', 'std','count'])
stats_pprocess['cv'] = stats_pprocess['std'] / stats_pprocess['mean']
data['perc_pprocess'] = data.groupby('parentProcessId')['timestamp'].rank(pct=True)

stats_user_event = data.groupby(['userId','eventId'])['timestamp'].agg(['mean', 'std','count'])
stats_user_event['cv'] = stats_user_event['std'] / stats_user_event['mean']
data['perc_user_event'] = data.groupby(['userId', 'eventId'])['timestamp'].rank(pct=True)

stats_user_process = data.groupby(['userId','processId'])['timestamp'].agg(['mean', 'std','count'])
stats_user_process['cv'] = stats_user_process['std'] / stats_user_process['mean']
data['perc_user_process'] = data.groupby(['userId', 'processId'])['timestamp'].rank(pct=True)

stats_user_pprocess = data.groupby(['userId','parentProcessId'])['timestamp'].agg(['mean', 'std','count'])
stats_user_pprocess['cv'] = stats_user_pprocess['std'] / stats_user_pprocess['mean']
data['perc_user_pprocess'] = data.groupby(['userId', 'parentProcessId'])['timestamp'].rank(pct=True)

stats_user_process_pprocess = data.groupby(['userId','processId','parentProcessId'])['timestamp'].agg(['mean', 'std','count'])
stats_user_process_pprocess['cv'] = stats_user_process_pprocess['std'] / stats_user_process_pprocess['mean']
data['perc_user_process_pprocess'] = data.groupby(['userId','processId','parentProcessId'])['timestamp'].rank(pct=True)

stats_event_process = data.groupby(['eventId','processId'])['timestamp'].agg(['mean', 'std','count'])
stats_event_process['cv'] = stats_event_process['std'] / stats_event_process['mean']
data['perc_event_process'] = data.groupby(['eventId','processId'])['timestamp'].rank(pct=True)

stats_event_pprocess = data.groupby(['eventId','parentProcessId'])['timestamp'].agg(['mean', 'std','count'])
stats_event_pprocess['cv'] = stats_event_pprocess['std'] / stats_event_pprocess['mean']
data['perc_event_pprocess'] = data.groupby(['eventId','parentProcessId'])['timestamp'].rank(pct=True)

stats_event_process_pprocess = data.groupby(['eventId','processId','parentProcessId'])['timestamp'].agg(['mean', 'std','count'])
stats_event_process_pprocess['cv'] = stats_event_process_pprocess['std'] / stats_event_process_pprocess['mean']
data['perc_event_process_pprocess'] = data.groupby(['eventId','processId','parentProcessId'])['timestamp'].rank(pct=True)

stats_process_pprocess = data.groupby(['processId','parentProcessId'])['timestamp'].agg(['mean', 'std','count'])
stats_process_pprocess['cv'] = stats_process_pprocess['std'] / stats_process_pprocess['mean']
data['perc_process_pprocess'] = data.groupby(['processId','parentProcessId'])['timestamp'].rank(pct=True)

stats_user_event_process_pprocess = data.groupby(['userId','eventId','processId','parentProcessId'])['timestamp'].agg(['mean', 'std','count'])
stats_user_event_process_pprocess['cv'] = stats_user_event_process_pprocess['std'] / stats_user_event_process_pprocess['mean']
data['perc_user_event_process_pprocess'] = data.groupby(['userId','eventId','processId','parentProcessId'])['timestamp'].rank(pct=True)

stats_user_processn = data.groupby(['userId','processName'])['timestamp'].agg(['mean', 'std','count'])
stats_user_processn['cv'] = stats_user_processn['std'] / stats_user_processn['mean']
data['perc_user_processn'] = data.groupby(['userId','processName'])['timestamp'].rank(pct=True)

stats_user_ip = data.groupby(['userId','ip'])['timestamp'].agg(['mean', 'std','count'])
stats_user_ip['cv'] = stats_user_ip['std'] / stats_user_ip['mean']
data['perc_user_ip'] = data.groupby(['userId','ip'])['timestamp'].rank(pct=True)

data = data.merge(stats_user,left_on='userId',suffixes=['','_user'],right_index=True)
data = data.merge(stats_event,left_on='eventId',suffixes=['','_event'],right_index=True)
data = data.merge(stats_process,left_on='processId',suffixes=['','_process'],right_index=True)
data = data.merge(stats_pprocess,left_on='parentProcessId',suffixes=['','_pprocess'],right_index=True)
data = data.merge(stats_user_event,left_on=['userId','eventId'],suffixes=['','_user_event'],right_index=True)
data = data.merge(stats_user_processn,left_on=['userId','processName'],suffixes=['','_user_processn'],right_index=True)
data = data.merge(stats_user_process,left_on=['userId','processId'],suffixes=['','_user_process'],right_index=True)
data = data.merge(stats_user_pprocess,left_on=['userId','parentProcessId'],suffixes=['','_user_pprocess'],right_index=True)
data = data.merge(stats_user_process_pprocess,left_on=['userId','processId','parentProcessId'],suffixes=['','_user_process_pprocess'],right_index=True)
data = data.merge(stats_event_process,left_on=['eventId','processId'],suffixes=['','_event_process'],right_index=True)
data = data.merge(stats_event_pprocess,left_on=['eventId','parentProcessId'],suffixes=['','_event_pprocess'],right_index=True)
data = data.merge(stats_event_process_pprocess,left_on=['eventId','processId','parentProcessId'],suffixes=['','_event_process_pprocess'],right_index=True)
data = data.merge(stats_process_pprocess,left_on=['processId','parentProcessId'],suffixes=['','_process_pprocess'],right_index=True)
data = data.merge(stats_user_event_process_pprocess,left_on=['userId','eventId','processId','parentProcessId'],suffixes=['','_user_event_process_pprocess'],right_index=True)
data = data.merge(stats_user_ip,left_on=['userId','ip'],suffixes=['','_user_ip'],right_index=True)

## probabilidades y entropias
stats_use_event = calc_stat(data,'userId','eventName')
stats_use_args = calc_stat(data,'userId','argsNum')
stats_use_pros = calc_stat(data,'userId','processName')
stats_use_ip2 = calc_stat(data,'userId','ip')

data = data.merge(stats_use_ip2, on=['userId','ip'], how='left',suffixes=['','_user_ip3'])

data['device'] = data['user-agent'].apply(extract_device)
stats_use_device = calc_stat(data,'userId','device')


#ip=[]
#network=[]
#version=[]
#city=[]
#region=[]
#region_code=[]
#country=[]
#country_name=[]
#country_code=[]
#country_code_iso3=[]
#country_capital=[]
#country_tld=[]
#continent_code=[]
#in_eu=[]
#postal=[]
#latitude=[]
#longitude=[]
#timezone=[]
#utc_offset=[]
#country_calling_code=[]
#currency=[]
#currency_name=[]
#languages=[]
#country_area=[]
#country_population=[]
#asn=[]
#org=[]
#sos=[]

#def scrape_ip_info(ip):
#    url = f"https://mxtoolbox.com/SuperTool.aspx?action=mx%3a{ip}&run=toolpage"

#    response = requests.get(url, verify=False)
#    soup = BeautifulSoup(response.content, "html.parser")

    # Extraer la información deseada de la página
#    result_table = soup.find("table", class_="resultTable")
#    if result_table:
#        rows = result_table.find_all("tr")
#        for row in rows:
#            cells = row.find_all("td")
#            if len(cells) == 2:
#                key = cells[0].text.strip()
#                value = cells[1].text.strip()
#                print(f"{key}: {value}")
                
#def get_ip_info(ip):
#    url = f"https://ipapi.co/{ip}/json/"
#    response = requests.get(url, verify=False)
#    data = response.json()
#    return data


#for i in data['ip'].unique():
#    ip_address = i
#    ip_info = get_ip_info(ip_address)
#    sospechoso = scrape_ip_info(ip)

# Imprimir información de la IP
#    ip.append(ip_info["ip"])
#    network.append(ip_info["network"])
#    version.append(ip_info["version"])
#    city.append(ip_info["city"])
#    region.append(ip_info["region"])
#    region_code.append(ip_info["region_code"])
#    country.append(ip_info["country"])
#    country_name.append(ip_info["country_name"])
#    country_code.append(ip_info["country_code"])
#    country_code_iso3.append(ip_info["country_code_iso3"])
#    country_capital.append(ip_info["country_capital"])
#    country_tld.append(ip_info["country_tld"])
#    continent_code.append(ip_info["continent_code"])
#    in_eu.append(ip_info["in_eu"])
#    postal.append(ip_info["postal"])
#    latitude.append(ip_info["latitude"])
#    longitude.append(ip_info["longitude"])
#    timezone.append(ip_info["timezone"])
#    utc_offset.append(ip_info["utc_offset"])
#    country_calling_code.append(ip_info["country_calling_code"])
#    currency.append(ip_info["currency"])
#    currency_name.append(ip_info["currency_name"])
#    languages.append(ip_info["languages"])
#    country_area.append(ip_info["country_area"])
#    country_population.append(ip_info["country_population"])
#    asn.append(ip_info["asn"])
#    org.append(ip_info["org"])
#    sos.append(sospechoso)

#    ips = pd.DataFrame({'ip':ip,
#        'network':network,
#        'version':version,
#        'city':city,
#        'region':region,
#        'region_code':region_code,
#        'country':country,
#        'country_name':country_name,
#        'country_code':country_code,
#        'country_code_iso3':country_code_iso3,
#        'country_capital':country_capital,
#        'country_tld':country_tld,
#        'continent_code':continent_code,
#        'in_eu':in_eu,
#        'postal':postal,
#        'latitude':latitude,
#        'longitude':longitude,
#        'timezone':timezone,
#        'utc_offset':utc_offset,
#        'country_calling_code':country_calling_code,
#        'currency':currency,
#        'currency_name':currency_name,
#        'languages':languages,
#        'country_area':country_area,
#        'country_population':country_population,
#        'asn':asn,
#        'org':org,
#        'sospechoso':sos})


ips = pd.read_csv(cfg['root_folder'] + "/data/"+"ips.csv")
data = pd.merge(data, ips, on=['ip'], how='left')

transactions = df_with_keywords['values']

# Instanciar y ajustar el codificador de transacciones
encoder = TransactionEncoder()

encoder_array = encoder.fit_transform(transactions)
df_encoded = pd.DataFrame(encoder_array, columns=encoder.columns_)

# Obtener los conjuntos de elementos frecuentes usando Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Obtener las reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence")

# Imprimir los conjuntos de elementos frecuentes y las reglas de asociación

#itemdf = pd.DataFrame(frequent_itemsets)
#def check_values(row):
#    supports = []
#    for item_set in row['values']:
#        for itemset in itemdf['itemsets']:
#            if item_set in list(itemset):
#                supports.append(itemdf.loc[itemdf['itemsets'] == itemset, 'support'].iloc[0])
#    return supports[0] if supports else 0
#df_with_keywords['support_values']=df_with_keywords.apply(check_values, axis=1)


val = ['ps',
'sshd',
'systemd',
'systemd-journal',
'amazon-ssm-agen',
'cron',
'snapd',
'systemd-resolve',
'systemd-tmpfile',
'systemd-network',
'systemd-logind',
'systemd-user-ru',
'ssm-agent-worke',
'accounts-daemon',
'(time-dir)',
'(tmpfiles)',
'bash',
'sh'
]
data['processName'] = data['processName'].apply(lambda x: x if x in val else 'otros')

data['returnValue'] = data['returnValue'].apply(lambda x: 1 if x == 0 else 0)
data = data.merge(stats_use_event, on=['userId','eventName'], how='left',suffixes=['','_user_event'],)


# Definir las listas

red = ['socket',
'connect',
'getsockname',
'accept4',
'bind',
'accept'
]
proc = [
'setreuid',
'cap_capable',
'prctl',
'execve',
'clone',
'kill',
'sched_process_exit',
'setuid',
'setregid',
'setgid',
'access',
'security_bprm_check',
]
arc = [
'openat',
'close',
'security_file_open',
'fstat',
'fchmod',
'stat',
'getdents64',
'unlink',
'dup3',
'dup2',
'dup',
'lstat',
'security_inode_unlink',
'unlinkat',
'umount',
'symlink'
]

# Crear la nueva columna
data['red'] = data['eventName'].isin(red).astype(int)
data['proc'] = data['eventName'].isin(proc).astype(int)
data['arc'] = data['eventName'].isin(arc).astype(int)
data = data.merge(stats_use_ip2, on=['userId','ip'], how='left',suffixes=['','_user_ip2'])


val2 = [
'security_file_open',
'openat',
'fstat',
'close',
'stat'
]
data['eventName'] = data['eventName'].apply(lambda x: x if x in val2 else 'otros')
data = pd.merge(data, stats_use_device, on=['userId','device'], how='left',suffixes=['','_user_device'])

data['dif_time_user'] = data['timestamp'] - data['mean']
data['dif_var_time_user'] = abs(data['dif_time_user']) - data['std']
data['dif_time_event'] = data['timestamp'] - data['mean_event']
data['dif_var_time_event'] = abs(data['dif_time_event']) - data['std_event']
data['dif_time_process'] = data['timestamp'] - data['mean_process']
data['dif_var_time_process'] = abs(data['dif_time_process']) - data['std_process']
data['dif_time_pprocess'] = data['timestamp'] - data['mean_pprocess']
data['dif_var_time_pprocess'] = abs(data['dif_time_pprocess']) - data['std_pprocess']
data['dif_time_user_event'] = data['timestamp'] - data['mean_user_event']
data['dif_var_time_user_event'] = abs(data['dif_time_user_event']) - data['std_user_event']
data['dif_time_user_processn'] = data['timestamp'] - data['mean_user_processn']
data['dif_var_time_user_processn'] = abs(data['dif_time_user_processn']) - data['std_user_processn']
data['dif_time_user_process'] = data['timestamp'] - data['mean_user_process']
data['dif_var_time_user_process'] = abs(data['dif_time_user_process']) - data['std_user_process']
data['dif_time_user_pprocess'] = data['timestamp'] - data['mean_user_pprocess']
data['dif_var_time_user_pprocess'] = abs(data['dif_time_user_pprocess']) - data['std_user_pprocess']
data['dif_time_user_process_pprocess'] = data['timestamp'] - data['mean_user_process_pprocess']
data['dif_var_time_user_process_pprocess'] = abs(data['dif_time_user_process_pprocess']) - data['std_user_process_pprocess']
data['dif_time_event_process'] = data['timestamp'] - data['mean_event_process']
data['dif_var_time_event_process'] = abs(data['dif_time_event_process']) - data['std_event_process']
data['dif_time_event_pprocess'] = data['timestamp'] - data['mean_event_pprocess']
data['dif_var_time_event_pprocess'] = abs(data['dif_time_event_pprocess']) - data['std_event_pprocess']
data['dif_time_event_process_pprocess'] = data['timestamp'] - data['mean_event_process_pprocess']
data['dif_var_time_event_process_pprocess'] = abs(data['dif_time_event_process_pprocess']) - data['std_event_process_pprocess']
data['dif_time_process_pprocess'] = data['timestamp'] - data['mean_process_pprocess']
data['dif_var_time_process_pprocess'] = abs(data['dif_time_process_pprocess']) - data['std_process_pprocess']
data['dif_time_user_event_process_pprocess'] = data['timestamp'] - data['mean_user_event_process_pprocess']
data['dif_var_time_user_event_process_pprocess'] = abs(data['dif_time_user_event_process_pprocess']) - data['std_user_event_process_pprocess']
data['dif_time_user_ip'] = data['timestamp'] - data['mean_user_ip']
data['dif_var_time_user_ip'] = abs(data['dif_time_user_ip']) - data['std_user_ip']

data['eventxuser'] = data['count_event'] / data['count']
data['processxuser'] = data['count_process'] / data['count']
data['pprocessxuser'] = data['count_pprocess'] / data['count']
data['ipxuser'] = data['count_user_ip'] / data['count']
data['usereventxuser'] = data['count_user_event'] / data['count']
data['userprocessnxuser'] = data['count_user_processn'] / data['count']
data['userprocessxuser'] = data['count_user_process'] / data['count']
data['userpprocessxuser'] = data['count_user_pprocess'] / data['count']
data['userprocesspprocessxuser'] = data['count_user_process_pprocess'] / data['count']
data['eventprocessxuser'] = data['count_event_process'] / data['count']
data['eventpprocessxuser'] = data['count_event_pprocess'] / data['count']
data['eventprocesspprocessxuser'] = data['count_event_process_pprocess'] / data['count']
data['processpprocessxuser'] = data['count_process_pprocess'] / data['count']
data['usereventprocesspprocessxuser'] = data['count_user_event_process_pprocess'] / data['count']
data = pd.merge(data, stats_use_pros, on=['userId','processName'], how='left',suffixes=['','_user_process'])
data = pd.merge(data, stats_use_args, on=['userId','argsNum'], how='left',suffixes=['','_user_arg'])

categorical_vars = ['eventName','processName','device','continent_code']
encoded_df=pd.get_dummies(data[categorical_vars])
numeric_vars = ['timestamp', 'argsNum','dif_time_user',
'dif_var_time_user',
'dif_time_event',
'dif_var_time_event',
'dif_time_process',
'dif_var_time_process',
'dif_time_pprocess',
'dif_var_time_pprocess',
'dif_time_user_event',
'dif_var_time_user_event',
'dif_time_user_processn',
'dif_var_time_user_processn',
'dif_time_user_process',
'dif_var_time_user_process',
'dif_time_user_pprocess',
'dif_var_time_user_pprocess',
'dif_time_user_process_pprocess',
'dif_var_time_user_process_pprocess',
'dif_time_event_process',
'dif_var_time_event_process',
'dif_time_event_pprocess',
'dif_var_time_event_pprocess',
'dif_time_event_process_pprocess',
'dif_var_time_event_process_pprocess',
'dif_time_process_pprocess',
'dif_var_time_process_pprocess',
'dif_time_user_event_process_pprocess',
'dif_var_time_user_event_process_pprocess',
'dif_time_user_ip',
'dif_var_time_user_ip','latitude','longitude']
scaler = MinMaxScaler()
scaled_vars = scaler.fit_transform(data[numeric_vars])
scaled_df = pd.DataFrame(scaled_vars, columns=numeric_vars).add_prefix('scaler_')

data = pd.concat([data, encoded_df, scaled_df], axis=1)


borrar = ['mean','std','utc_offset', 'org', 'count','city','latitude','longitude',
       'region', 'country_name', 'continent_code', 'in_eu','mean_user_ip', 'std_user_ip',
       'count_user_ip','mean_user_event_process_pprocess',
       'std_user_event_process_pprocess', 'count_user_event_process_pprocess','mean_process_pprocess', 'std_process_pprocess',
       'count_process_pprocess','mean_event_process_pprocess', 'std_event_process_pprocess',
       'count_event_process_pprocess','mean_event_pprocess','std_event_pprocess', 'count_event_pprocess','mean_event_process', 'std_event_process',
       'count_event_process','mean_user_process_pprocess',
       'std_user_process_pprocess', 'count_user_process_pprocess','mean_user_pprocess', 'std_user_pprocess',
       'count_user_pprocess','mean_user_process', 'std_user_process', 'count_user_process','std_user_processn', 
       'count_user_processn','mean_user_processn','mean_user_event', 'std_user_event', 'count_user_event','mean_pprocess',
        'std_pprocess', 'count_pprocess','mean_process', 'std_process', 'count_process','mean_event','std_event', 'count_event',
        'processId','parentProcessId','userId','eventId','hostName','sus','evil','args','user-agent','ip','values5',
        'argsNum','dif_time_user','device_Linux',
        'dif_var_time_user',
        'dif_time_event',
        'dif_var_time_event',
        'dif_time_process',
        'dif_var_time_process',
        'dif_time_pprocess',
        'dif_var_time_pprocess',
        'dif_time_user_event',
        'dif_var_time_user_event',
        'dif_time_user_processn',
        'dif_var_time_user_processn',
        'dif_time_user_process',
        'dif_var_time_user_process',
        'dif_time_user_pprocess',
        'dif_var_time_user_pprocess',
        'dif_time_user_process_pprocess',
        'dif_var_time_user_process_pprocess',
        'dif_time_event_process',
        'dif_var_time_event_process',
        'dif_time_event_pprocess',
        'dif_var_time_event_pprocess',
        'dif_time_event_process_pprocess',
        'dif_var_time_event_process_pprocess',
        'dif_time_process_pprocess',
        'dif_var_time_process_pprocess',
        'dif_time_user_event_process_pprocess',
        'dif_var_time_user_event_process_pprocess',
        'dif_time_user_ip',
        'dif_var_time_user_ip','eventName','processName','device','continent_code',
        'min_probability_user_valuearg',
        'userprocesspprocessxuser',
        'eventprocessxuser',
        'eventpprocessxuser',
        'processpprocessxuser',
        'usereventprocesspprocessxuser'
]

data = data.drop(borrar, axis=1).fillna(0)

data['id'] = range(1, len(data) + 1)

data2 = data.copy()
data2 = data2.drop(['entropy_user_valuearg',
'cv',
'cv_event',
'cv_process',
'cv_pprocess',
'cv_user_event',
'cv_user_processn',
'cv_user_process',
'cv_user_pprocess',
'cv_user_process_pprocess',
'cv_event_process',
'cv_event_pprocess',
'cv_event_process_pprocess',
'cv_process_pprocess',
'cv_user_event_process_pprocess',
'cv_user_ip',
'entropy',
'entropy_user_event',
'entropy_user_arg',
'entropy_user_process',
'entropy_user_device',
'entropy_user_ip2',
'processName_(time-dir)',
'processName_(tmpfiles)',
'processName_accounts-daemon',
'processName_bash',
'processName_otros',
'processName_sh',
'processName_ssm-agent-worke',
'processName_systemd-logind',
'processName_systemd-user-ru'
], axis=1).fillna(0)

data3 = data2.copy()
data3 = data3.drop([
'eventxuser',
'processxuser',
'pprocessxuser',
'ipxuser',
'usereventxuser',
'userprocessnxuser',
'userprocessxuser',
'userpprocessxuser',
'eventprocesspprocessxuser'], axis=1).fillna(0)

data.to_csv(cfg['root_folder'] + "/data/"+'data_model.csv', index=False)
data2.to_csv(cfg['root_folder'] + "/data/"+'data_model2.csv', index=False)
data3.to_csv(cfg['root_folder'] + "/data/"+'data_model3.csv', index=False)
