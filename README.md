# Detección de Anomalías
## Marisol Correa Henao


Se ha realizado el proceso bajo el framework CRISP-DM (Cross-Industry Standard Process for Data Mining), por lo que se divide el proceso en 
- **Comprensión del negocio:** Se analiza y se entiende el contexto UEBA "User and Entity Behavior Analytics" y se revisan trabajos previos en este contexto y en otras áreas
- **Comprensión de los datos:** Se explora y analiza los datos disponibles para comprender su calidad, estructura y características, se revisa cada variable disponibles, su tipo, su valor y qué significa dentro del contexto UEBA se estudia cada categoría no sólo para tener conocimeinto del problema sino para crear nuevas variables a partir de ese conocimiento.
Ejemplo:

Categorias Event
socket: Creación y manipulación de sockets de red.
openat: Apertura de archivos o directorios utilizando una ruta relativa a un descriptor de archivo.
close: Cierre de un descriptor de archivo.
security_file_open: Verificación de seguridad al abrir un archivo.
fstat: Obtención de información sobre un archivo utilizando un descriptor de archivo.
connect: Establecimiento de una conexión de red a un host remoto.
fchmod: Cambio de los permisos de un archivo utilizando un descriptor de archivo.
stat: Obtención de información sobre un archivo utilizando su ruta.
setreuid: Cambio del UID real y efectivo del proceso.
getsockname: Obtención de la dirección asociada a un socket local.
getdents64: Lectura de las entradas de un directorio.
unlink: Eliminación de un archivo.
cap_capable: Verificación de capacidad de capacidad de proceso.
prctl: Control y configuración de diferentes aspectos del proceso.
dup3: Duplicación de un descriptor de archivo, especificando un descriptor específico.
execve: Ejecución de un nuevo programa en el proceso actual.
clone: Creación de un nuevo proceso o hilo.
accept4: Aceptación de una conexión de red entrante en un socket.
kill: Envío de una señal a un proceso.
access: Verificación de los permisos de acceso a un archivo.
dup2: Duplicación de un descriptor de archivo, especificando un descriptor de destino.
security_bprm_check: Verificación de seguridad al ejecutar un programa.
dup: Duplicación de un descriptor de archivo.
sched_process_exit: Planificación de la salida de un proceso.
setuid: Cambio del UID efectivo del proceso.
lstat: Obtención de información sobre un archivo o enlace simbólico utilizando su ruta.
setregid: Cambio del GID real y efectivo del proceso.
bind: Asociación de un socket a una dirección IP y puerto local.
security_inode_unlink: Verificación de seguridad al eliminar un archivo.
accept: Aceptación de una conexión de red entrante en un socket.
setgid: Cambio del GID efectivo del proceso.
unlinkat: Eliminación de un archivo o enlace simbólico utilizando un descriptor de archivo y una ruta relativa.
umount: Desmontaje de un sistema de archivos.
symlink: Creación de un enlace simbólico.

categorización: red,proceso,archivos

*Este proceso se encuentra en el archivo describe.ipynb en la carpeta notebooks*

- **Preparación de los datos:** Se realizan algunas correcciones a los datos, limpieza, imputación y se eliminan variables que no generan valor al no aportar ninguna variabilidad al modelo, se identifican las variables relevantes para la detección de anomalías y comprende las relaciones entre éstas y posibles variables a obtener.
Atravpes de diferentes técnicas se obtienen variables nuevas:
    + transformaciones a variables actuales de manera que aportaran más información o pudiera ser más intuitivo para el modelo encontrar anomalía a partir de ellas como top process o top event.
    + creación de variables nuevas partiendo de estadísticos conocidos y las variables actuales como medidas de dispersión, media, percentiles y la entropia que mide la incertidumbre o la aleatoriedad en la distribución de eventos y puede ayudar a identificar usuarios con comportamientos más impredecibles o atípicos, cuanto mayor sea la entropía, mayor será la incertidumbre y mayor será la cantidad de información necesaria para describir los datos.
    + Técnicas de análisis de texto para extraer información adicional a través de tokenización de texto.
    + Modelo de detección de anomalías desde la creación de variables y no limitadas al modelo final a través del modelo apriori de association rules.
    + Técnicas de extracción de información externa contenida en api, como el caso de las ips se tiene una api para detectar si es sospechosa o no y otra para extraer info general de la ip como geolocalización, dispositivo, región, etc.
    + Técnicas de transformación de variables de algún tipo especial.
    + Patrones subyacentes en los datos.
    
Se Realiza transformaciones de datos, como normalización o codificación de variables categóricas de manera que los datos queden en los formatos solicitados por cada modelo generando un ETL con 378.424 observaciones 143 variables.

*Este proceso se encuentra en el archivo datos.py en la carpeta src*

- **Modelado:** Se implementa y ajusta diferentes modelos de detección de anomalías bajo tres conjuntos diferentes de variables y 3 escenarios de parámetros diferentes, los modelos usados fueron:
    + K-means: se utilizan los clústeres generados por el algoritmo K-means. Si una observación se encuentra en un clúster con una densidad de puntos significativamente menor en comparación con otros clústeres, se considera anómala.

    + Autoencoders: Los autoencoders son entrenados para reconstruir los datos de entrada. La puntuación de anomalía se calcula comparando la diferencia entre la entrada original y la salida reconstruida del autoencoder. Cuanto mayor sea la diferencia, más anómala se considera la observación.

    + One-Class SVM: Este modelo se entrena en un conjunto de datos que se considera representativo de la clase normal. Las observaciones que se alejan significativamente de este conjunto se consideran anómalas.

    + Isolation Forest: Este algoritmo crea un bosque de árboles de aislamiento que dividen el conjunto de datos en subconjuntos más pequeños. Si una observación requiere menos divisiones para ser aislada, se considera anómala.

    + Association Rules: Se utiliza el algoritmo de reglas de asociación para identificar combinaciones inusuales de variables. Si una observación contiene una combinación de variables que no es común según las reglas de asociación aprendidas, se considera anómala.

Se analiza y contrasta resultados a través del análisis de componente principales (PCA).

*Este proceso se encuentra en el archivo model.py en la carpeta src*

- **Evaluación:** Se evalúa el rendimiento de los modelos utilizando métricas adecuadas para la detección de anomalías. Examina y compara los resultados de los diferentes modelos para seleccionar el mejor.
Es importante tener en cuenta que en el aprendizaje no supervisado, la evaluación puede ser más subjetiva y depende en gran medida del conocimiento y la interpretación del problema. Por lo tanto, a través del conocimiento experto se toman decisiones como análizar mejores modelos y hacer merge de resultados para generar una mejor tasa de detección.
Ejemplo:
    + Decision function: Puntuación de anomalía de una instancia
    + Silhouette: Homogeneidad intra cluster y heterogeneidad entre cluster
    + Soporte: El número de instancias asignadas a un grupo o cluster específico.
    + Confidence: Mide la probabilidad condicional de que el consecuente ocurra dado que el antecedente ha ocurrido.
    + lift: Mide la fuerza de la asociación entre el antecedente y el consecuente, en comparación con si fueran eventos independientes.
    + Leverage: La ganancia, mide la diferencia entre la frecuencia observada de la ocurrencia conjunta del antecedente y el consecuente, y la frecuencia esperada si fueran eventos independientes.
    + Conviction: Mide la dependencia del consecuente dado el antecedente, teniendo en cuenta la frecuencia observada y esperada.
    + métrica de Zhang: Medida de calidad utilizada en el algoritmo Apriori para evaluar la importancia de los conjuntos de elementos frecuentes. 

Luego de aplicar las métricas y seleccionar los mejores modelos se toman las observaciones anómalas y se ajustan para obtener las más veridícas y obtener una mejor identificación.

*Este análisis se encuentra en el archivo resultados.ipynb en la carpeta notebooks*

- **Despliegue:** Se Implementa el modelo seleccionado en producción a través de aws con una lambda function y una api gateway que hace llamados a la función lambda, se realiza pruebas y llamados en nube para asegurar que funcione correctamente en entornos reales.

- **Ciclo iterativo:** El proceso CRISP-DM es iterativo y cíclico por lo que permite hacer mejoras, se proponen las siguientes:

    + Inferencias sobre timestamp: tener el campo tipo fecha permitiría explorar variables adicionales como tiempo entre eventos, dias o fechas de recurrencia, eentre otras que aportarían más información al modelo e incluso permitirían ver anomalias a través de series de tiempo.
    + Profundizar en Argumentos
    + Transformaciones u otros modelos
    + Mejor Procesamiento: implementar autoencoder en la nube
    + Aprovechamiento Association rules: Aplicar este modelo a las demás variables categóricas relevantes



*todos los conjuntos de datos generados en cada proceso se encuentran en la carpeta data*

*En la carpeta notebook se encuentra documentación adicional*

*Adicional: En la carpeta src/app se encuentra un archivo app.py para consumir el modelo a través de la app*
*Adicional: En la carpeta src/app se encuentra un archivo call.py para hacer peticiones a la api rest de api gateway de aws y hacer solicitudes para saber si una observación es anómala*


**El archivo con la puntuación final resultado del modelo se encuentra en la carpeta /src con el nombre result.csv, la columna anomaly_final -1 indica que NO hay anomalía, el restante se consideran anomalas siendo 1 y 2 categoría de anomalía fuerte**