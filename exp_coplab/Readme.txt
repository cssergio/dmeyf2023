0: Para cada corrida se debe modificar el nombre del experimento
1: Correr el z823 con PARAM$trainingstrategy$undersampling <- 1.0
2: Correr el z823 con PARAM$trainingstrategy$undersampling <- 0.5
3: Correr el z823 con PARAM$trainingstrategy$undersampling <- 0.3
4: Correr el z823 con PARAM$trainingstrategy$undersampling <- 0.1
5: Para cada BO_log generado se ordena por ganancia y se registran los hiperparametros que generaron en cada uno la mayor ganancia.
6: Se corre el script z824 con cada configuracion de hiperparametros.
7: El z824 realiza el modelo final 20 veces cambiando la semilla.
