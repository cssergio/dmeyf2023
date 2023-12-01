Primero correr clase_ternaria_mp.r para crear a partir de competencia_03_crudo la clase ternaria.
Segundo correr CA_FE.r que ancara las variables rotas y la creación de variables historicas lag y delta 1,2 y 6 y media 3 y 6.
Tercero correr z823_bayesiana_goss_us01.r para realizar la optimización bayesiana y asi obtener los hiperparámetros que utilizaremos en el script final.
Cuarto corremos z824_final_goss.r con los mejores HPs que arrojó el punto anterior que son:
num_iterations <- 508
learning_rate <- 0.14113801
min_data_in_leaf <- 4875
num_leaves <- 68
feature_fraction_bynode <- 0.885693535
max_depth <- 36
top_rate <- 0.325814945
other_rate <- 0.307901019
feature_fraction <- 0.662296515444177
Este último script realiza un semillerío con 20 semillas promediando por numero de cliente las probabilidades que arroja la predicción del modelo para 202109, luego se ordenan de mayor a menor y se realiza el envío a los primeros 10500 clientes.
Tambien está subida la bayesiana obtenida luego de correr el script z823_bayesiana_goss_us01.r y el archivo para subir a kaggle generado por el script z823_finsl_goss.r para 10500 envios de corte.
