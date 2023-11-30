require("data.table")

setwd("~/buckets/b1/") # Establezco el Working Directory
dataset <- fread("./datasets/competencia_03.csv.gz")
# datset$tmobile_app <- NULL

setorder( dataset, numero_de_cliente, foto_mes)

dataset[foto_mes == 201901, ctransferencias_recibidas := NA]
dataset[foto_mes == 201901, mtransferencias_recibidas := NA ]

dataset[foto_mes == 201902, ctransferencias_recibidas := NA]
dataset[foto_mes == 201902, mtransferencias_recibidas := NA]

dataset[foto_mes == 201903, ctransferencias_recibidas := NA]
dataset[foto_mes == 201903, mtransferencias_recibidas := NA]

dataset[foto_mes == 201904, ctarjeta_visa_debitos_automaticos := NA]
dataset[foto_mes == 201904, ctransferencias_recibidas := NA]
dataset[foto_mes == 201904, mtransferencias_recibidas := NA]
dataset[foto_mes == 201904, mttarjeta_visa_debitos_automaticos := NA]
dataset[foto_mes == 201904, Visa_mfinanciacion_limite := NA]

dataset[foto_mes == 201905, ccomisiones_otras := NA]
dataset[foto_mes == 201905, ctarjeta_visa_debitos_automaticos := NA]
dataset[foto_mes == 201905, ctransferencias_recibidas := NA]
dataset[foto_mes == 201905, mactivos_margen := NA]
dataset[foto_mes == 201905, mcomisiones := NA]
dataset[foto_mes == 201905, mcomisiones_otras := NA]
dataset[foto_mes == 201905, mpasivos_margen := NA]
dataset[foto_mes == 201905, mrentabilidad_annual := NA]
dataset[foto_mes == 201905, mrentabilidad := NA]
dataset[foto_mes == 201905, mtransferencias_recibidas := NA]

dataset[foto_mes == 201910, ccajeros_propios_descuentos := NA]
dataset[foto_mes == 201910, ccomisiones_otras := NA]
dataset[foto_mes == 201910, chomebanking_transacciones := NA]
dataset[foto_mes == 201910, ctarjeta_master_descuentos := NA]
dataset[foto_mes == 201910, ctarjeta_visa_descuentos := NA]
dataset[foto_mes == 201910, mactivos_margen := NA]
dataset[foto_mes == 201910, mcajeros_propios_descuentos := NA]
dataset[foto_mes == 201910, mcomisiones := NA]
dataset[foto_mes == 201910, mcomisiones_otras := NA]
dataset[foto_mes == 201910, mpasivos_margen := NA]
dataset[foto_mes == 201910, mrentabilidad_annual := NA]
dataset[foto_mes == 201910, mrentabilidad := NA]
dataset[foto_mes == 201910, mtarjeta_master_descuentos := NA]
dataset[foto_mes == 201910, mtarjeta_visa_descuentos := NA]

dataset[foto_mes == 202001, cliente_vip := NA]

dataset[foto_mes == 202006, active_quarter := NA]
dataset[foto_mes == 202006, catm_trx := NA]
dataset[foto_mes == 202006, catm_trx_other := NA]
dataset[foto_mes == 202006, ccajas_consultas := NA]
dataset[foto_mes == 202006, ccajas_depositos := NA]
dataset[foto_mes == 202006, ccajas_extracciones := NA]
dataset[foto_mes == 202006, ccajas_otras := NA]
dataset[foto_mes == 202006, ccajas_transacciones := NA]
dataset[foto_mes == 202006, ccallcenter_transacciones := NA]
dataset[foto_mes == 202006, ccheques_depositados := NA]
dataset[foto_mes == 202006, ccheques_depositados_rechazados := NA]
dataset[foto_mes == 202006, ccheques_emitidos := NA]
dataset[foto_mes == 202006, ccheques_emitidos_rechazados := NA]
dataset[foto_mes == 202006, ccomisiones_otras := NA]
dataset[foto_mes == 202006, cextraccion_autoservicio := NA]
dataset[foto_mes == 202006, chomebanking_transacciones := NA]
dataset[foto_mes == 202006, cmobile_app_trx := NA]
dataset[foto_mes == 202006, ctarjeta_debito_transacciones := NA]
dataset[foto_mes == 202006, ctarjeta_master_transacciones := NA]
dataset[foto_mes == 202006, ctarjeta_visa_transacciones := NA]
dataset[foto_mes == 202006, ctrx_quarter := NA]
dataset[foto_mes == 202006, mactivos_margen := NA]
dataset[foto_mes == 202006, matm := NA]
dataset[foto_mes == 202006, matm_other := NA]
dataset[foto_mes == 202006, mautoservicio := NA]
dataset[foto_mes == 202006, mcheques_depositados := NA]
dataset[foto_mes == 202006, mcheques_depositados_rechazados := NA]
dataset[foto_mes == 202006, mcheques_emitidos := NA]
dataset[foto_mes == 202006, mcheques_emitidos_rechazados := NA]
dataset[foto_mes == 202006, mcomisiones := NA]
dataset[foto_mes == 202006, mcomisiones_otras := NA]
dataset[foto_mes == 202006, mcuentas_saldo := NA]
dataset[foto_mes == 202006, mextraccion_autoservicio := NA]
dataset[foto_mes == 202006, mpasivos_margen := NA]
dataset[foto_mes == 202006, mrentabilidad_annual := NA]
dataset[foto_mes == 202006, mrentabilidad := NA]
dataset[foto_mes == 202006, mtarjeta_master_consumo := NA]
dataset[foto_mes == 202006, mtarjeta_visa_consumo := NA]
dataset[foto_mes == 202006, tcallcenter := NA]
dataset[foto_mes == 202006, thomebanking := NA]

cols_lagueables <- copy(setdiff( colnames(dataset), c("numero_de_cliente", "foto_mes", "clase_ternaria") ))

# lags de orden 1
dataset[, paste0(cols_lagueables, "_lag1") := shift(.SD, 1L, NA, "lag"), by = numero_de_cliente, .SDcols = cols_lagueables]

# agrego los delta lags de orden 1
for (vcol in cols_lagueables) dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]

# lags de orden 2
dataset[, paste0(cols_lagueables, "_lag2") := shift(.SD, 2L, NA, "lag"), by = numero_de_cliente, .SDcols = cols_lagueables]

# agrego los delta lags de orden 2
for (vcol in cols_lagueables) dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]

# lags de orden 6
dataset[, paste0(cols_lagueables, "_lag6") := shift(.SD, 6L, NA, "lag"), by = numero_de_cliente, .SDcols = cols_lagueables]

# agrego los delta lags de orden 6
for (vcol in cols_lagueables) dataset[, paste0(vcol, "_delta6") := get(vcol) - get(paste0(vcol, "_lag6"))]

### Media móvil 3 meses
dataset[,paste0(cols_lagueables, "_avg3") := frollmean(.SD, 3, na.rm=T), .SDcols = cols_lagueables, by = numero_de_cliente]

### Media móvil 6 meses
dataset[,paste0(cols_lagueables, "_avg6") := frollmean(.SD, 6, na.rm=T), .SDcols = cols_lagueables, by = numero_de_cliente]

fwrite(dataset, file = "./datasets/competencia_03_CA_FE.csv.gz", sep = ",")

