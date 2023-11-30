# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM
# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")
require('dplyr')
library(dplyr)

# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento <- "experimentos_mm_con_cafe_2_goss_final"

PARAM$input$dataset <- "./datasets/competencia_03_CA_FE.csv.gz"

# meses donde se entrena el modelo.
# roll forward un mes

PARAM$input$training <- c(201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912, 202001, 202002, 202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202106, 202107)
PARAM$input$future <- c(202109) # meses donde se aplica el modelo

# un undersampling de 0.1  toma solo el 10% de los CONTINUA
PARAM$trainingstrategy$undersampling <- 1
PARAM$trainingstrategy$semilla_azar <- 102191 # Aqui poner su  primer  semilla

semillas <- c(550007, 550009, 550027, 550049, 550051,550067, 550073, 550111, 550117, 550127, 550129, 550139, 550163, 550169, 550177, 550181, 550189, 550199, 550211, 550213)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------

# defino los datos que forma parte del training
# aqui se hace el undersampling de los CONTINUA
set.seed(PARAM$trainingstrategy$semilla_azar)
dataset[, azar := runif(nrow(dataset))]
dataset[, train := 0L]
dataset[
  foto_mes %in% PARAM$input$training &
	(azar <= PARAM$trainingstrategy$undersampling | clase_ternaria %in% c("BAJA+1", "BAJA+2")),
  train := 1L
]


#--------------------------------------
# creo las carpetas donde van los resultados
# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))

# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)

# Obtengo los datos a predecir
dapply <- dataset[foto_mes == PARAM$input$future]

# Selecciono columna con numero de cliente y foto mes en df para guardar las predicciones
predicciones <- dapply[, list(numero_de_cliente, foto_mes)]

for (i in 1:20) {
  
	PARAM$finalmodel$semilla <- semillas[i]		
	 
	# hiperparametros intencionalmente 
	PARAM$finalmodel$optim$num_iterations <- 508
	PARAM$finalmodel$optim$learning_rate <- 0.14113801
	PARAM$finalmodel$optim$min_data_in_leaf <- 4875
	PARAM$finalmodel$optim$num_leaves <- 68
	PARAM$finalmodel$optim$feature_fraction_bynode <- 0.885693535
	PARAM$finalmodel$optim$max_depth <- 36
	PARAM$finalmodel$optim$top_rate <- 0.325814945
	PARAM$finalmodel$optim$other_rate <- 0.307901019
	PARAM$finalmodel$optim$feature_fraction <- 0.662296515444177

	# Hiperparametros FIJOS de  lightgbm
	PARAM$finalmodel$lgb_basicos <- list(
	boosting = "gbdt", # puede ir  dart  , ni pruebe random_forest
	objective = "binary",
	metric = "custom",
	first_metric_only = TRUE,
	boost_from_average = TRUE,
	feature_pre_filter = FALSE,	
	force_row_wise = TRUE, # para reducir warnings
	verbosity = -100,
	max_depth = -1L, # -1 significa no limitar,  por ahora lo dejo fijo
	min_gain_to_split = 0.0, # min_gain_to_split >= 0.0
	min_sum_hessian_in_leaf = 0.001, #  min_sum_hessian_in_leaf >= 0.0
	lambda_l1 = 0.0, # lambda_l1 >= 0.0
	lambda_l2 = 0.0, # lambda_l2 >= 0.0
	max_bin = 31L, # lo debo dejar fijo, no participa de la BO

	bagging_fraction = 1.0, # 0.0 < bagging_fraction <= 1.0
	pos_bagging_fraction = 1.0, # 0.0 < pos_bagging_fraction <= 1.0
	neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
	is_unbalance = FALSE, #
	scale_pos_weight = 1.0, # scale_pos_weight > 0.0

	drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
	max_drop = 50, # <=0 means no limit
	skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0

	extra_trees = FALSE, # Magic Sauce

	seed = PARAM$finalmodel$semilla
	)

	param_completo <- c(PARAM$finalmodel$lgb_basicos,
					  PARAM$finalmodel$optim)


	modelo <- lgb.train(
	data = dtrain,
	param = param_completo,
	)

	#--------------------------------------
	# ahora imprimo la importancia de variables
	tb_importancia <- as.data.table(lgb.importance(modelo))
	archivo_importancia <- paste0("impo_",i,".txt")

	fwrite(tb_importancia,
		 file = archivo_importancia,
		 sep = "\t"
	)

	#--------------------------------------


	# aplico el modelo a los datos sin clase
	dapply <- dataset[foto_mes == PARAM$input$future]

	# aplico el modelo a los datos nuevos
	prediccion <- predict(
	modelo,
	data.matrix(dapply[, campos_buenos, with = FALSE])
	)

	# genero la tabla de entrega
	tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
	tb_entrega[, prob := prediccion]

	# grabo las probabilidad del modelo
	fwrite(tb_entrega,
		 file = paste0("prediccion_",i,".txt"),
		 sep = "\t"
	)

	# ordeno por probabilidad descendente
	setorder(tb_entrega, -prob)


	# Agrego columna con las predicciones de cada semilla
	col_name <- paste0("semilla_", semillas[i])
	predicciones[, (col_name) := prediccion] 

	print(paste0("Iteracion ",i, " finalizada"))
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")

#-------------------------------PERSISTO SALIDA CON LAS PREDICCIONES DE CADA SEMILLA------------------------------#

# Guardo el archivo
archivo_salida <- paste0(PARAM$experimento, "_predicciones_semillas.csv")
fwrite(predicciones, file = archivo_salida, sep = ",")

# Calcular el promedio de las predicciones para cada cliente
predicciones_promedio <- predicciones[, .(promedio = rowMeans(.SD, na.rm = TRUE)), by = numero_de_cliente, .SDcols = paste0("semilla_", semillas)]

# Ordenar el dataframe por el promedio en orden decreciente
predicciones_promedio <- predicciones_promedio[order(-promedio)]

# Realizar operaciones adicionales (aquí es donde puedes ajustar según tus necesidades)
cortes <- c(seq(8000, 15000, by = 500))

for (envios in cortes) {
	# Configurar la columna 'Predicted' en función de ciertas condiciones (ajustar según tus necesidades)
	predicciones_promedio[, Predicted := 0L]
	predicciones_promedio[1:envios, Predicted := 1L]

	# Guardar el resultado en un archivo CSV
	fwrite(predicciones_promedio[, list(numero_de_cliente, Predicted)],
		 file = paste0(PARAM$experimento, "_", envios, ".csv"),
		 sep = ",")
}
