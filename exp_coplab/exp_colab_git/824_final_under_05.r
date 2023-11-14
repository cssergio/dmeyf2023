# Cargar bibliotecas
require("data.table")
require("lightgbm")

# Definir semillas primas a partir de 17
semillas <- c(550007, 550009, 550027, 550049, 550051, 550067, 550073, 550111, 550117, 550127, 550129, 550139, 550163, 550169, 550177, 550181, 550189, 550199, 550211, 550213)

# Leer el conjunto de datos una vez fuera del ciclo
# Cambiar directorio de trabajo
setwd("~/buckets/b1")

dataset <- fread("./datasets/dataset_baseline_exp_colab.csv.gz", stringsAsFactors = TRUE)
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))
dataset[, train := 0L]
dataset[foto_mes %in% c(202101, 202102, 202103, 202104, 202105, 202106), train := 1L]

# Configurar el directorio de trabajo fijo
dir.create("./exp/KA8240_under_05_exp_colab/", showWarnings = FALSE)
setwd("./exp/KA8240_under_05_exp_colab/")

# Configurar parámetros comunes
PARAM <- list()
PARAM$input$future <- 202107
PARAM$finalmodel$optim$num_iterations <- 565
PARAM$finalmodel$optim$learning_rate <- 0.213915727
PARAM$finalmodel$optim$feature_fraction <- 0.858676559
PARAM$finalmodel$optim$min_data_in_leaf <- 46159
PARAM$finalmodel$optim$num_leaves <- 1018


PARAM$finalmodel$lgb_basicos <- list(
  boosting = "gbdt", objective = "binary", metric = "custom",
  first_metric_only = TRUE, boost_from_average = TRUE, feature_pre_filter = FALSE,
  force_row_wise = TRUE, verbosity = -100, max_depth = -1L,
  min_gain_to_split = 0.0, min_sum_hessian_in_leaf = 0.001, lambda_l1 = 0.0, lambda_l2 = 0.0,
  max_bin = 31L, bagging_fraction = 1.0, pos_bagging_fraction = 1.0,
  neg_bagging_fraction = 1.0, is_unbalance = FALSE, scale_pos_weight = 1.0,
  drop_rate = 0.1, max_drop = 50, skip_drop = 0.5, extra_trees = TRUE
)

# Iterar sobre las semillas
for (semilla in semillas) {
  # Configurar la semilla
  PARAM$finalmodel$semilla <- semilla
  PARAM$finalmodel$lgb_basicos$seed <- PARAM$finalmodel$semilla
  # Crear dataset de entrenamiento
  dtrain <- lgb.Dataset(data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
                        label = dataset[train == 1L, clase01])
  
  # Entrenar el modelo
  param_completo <- c(PARAM$finalmodel$lgb_basicos, PARAM$finalmodel$optim)
  modelo <- lgb.train(data = dtrain, param = param_completo)
  
  # Aplicar el modelo a los datos de prueba
  dapply <- dataset[foto_mes == PARAM$input$future]
  prediccion <- predict(modelo, data.matrix(dapply[, campos_buenos, with = FALSE]))
  
  # Generar la tabla de entrega
  tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
  tb_entrega[, prob := prediccion]
  
  # Guardar la predicción en un archivo específico para cada semilla
  nombre_archivo <- paste0("prediccion_", semilla, ".txt")
  fwrite(tb_entrega, file = nombre_archivo, sep = "\t")
  
  cat("\n\nLa generación de los archivos para la semilla ", semilla, " ha terminado\n")
}

