# Resultados — HeartlyPE L1 (protocolo corregido)

Todas las cifras salen de corridas con log (`reports/cv/*/log_*.txt`, `reports/oof/*.npz`, `reports/test/*.md`). Regla de decisión: **argmax** (umbral SVEB elegido por OOF; ver `reports/thresholds/`). Media de 3 semillas [mín–máx]; el ensamble promedia las probabilidades de las 3 semillas.

## Validación cruzada (StratifiedGroupKFold k=5 por **sujeto real**, 66 sujetos MIT-BIH+INCART, early stopping con validación interna por sujeto)

| modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|
| E12 (3 semillas) | 0.637 [0.600–0.699] | 0.283 [0.159–0.437] | 0.232 [0.135–0.362] | 0.255 [0.146–0.396] | 0.798 [0.758–0.853] | 0.625 [0.577–0.650] | 0.700 [0.665–0.738] |
| E12 ensamble 3 | 0.647 | 0.279 | 0.242 | 0.260 | 0.809 | 0.647 | 0.719 |
| E13_rrctx (3 semillas) | 0.690 [0.674–0.707] | 0.485 [0.439–0.532] | 0.385 [0.349–0.452] | 0.427 [0.391–0.468] | 0.780 [0.731–0.839] | 0.612 [0.591–0.622] | 0.685 [0.672–0.694] |
| E13_rrctx ensamble 3 | 0.709 | 0.510 | 0.415 | 0.457 | 0.793 | 0.638 | 0.707 |
| E14_incII (3 semillas) | 0.744 [0.735–0.752] | 0.584 [0.550–0.605] | 0.357 [0.341–0.366] | 0.443 [0.421–0.454] | 0.882 [0.880–0.883] | 0.760 [0.746–0.779] | 0.816 [0.809–0.827] |
| E14_incII ensamble 3 | 0.759 | 0.585 | 0.384 | 0.464 | 0.891 | 0.789 | 0.837 |

### Solo registros MIT-BIH del pool (37 sujetos; comparable con la literatura)

| modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|
| E12 (3 semillas) | 0.587 [0.546–0.652] | 0.212 [0.115–0.326] | 0.241 [0.083–0.403] | 0.224 [0.096–0.360] | 0.859 [0.826–0.886] | 0.457 [0.392–0.504] | 0.595 [0.540–0.643] |
| E12 ensamble 3 | 0.587 | 0.195 | 0.220 | 0.207 | 0.868 | 0.468 | 0.608 |
| E13_rrctx (3 semillas) | 0.619 [0.599–0.632] | 0.346 [0.309–0.388] | 0.347 [0.294–0.421] | 0.342 [0.316–0.356] | 0.814 [0.752–0.864] | 0.440 [0.425–0.463] | 0.571 [0.543–0.593] |
| E13_rrctx ensamble 3 | 0.629 | 0.350 | 0.368 | 0.359 | 0.827 | 0.449 | 0.582 |
| E14_incII (3 semillas) | 0.664 [0.645–0.678] | 0.408 [0.378–0.447] | 0.259 [0.211–0.284] | 0.316 [0.271–0.346] | 0.820 [0.815–0.823] | 0.633 [0.620–0.644] | 0.715 [0.707–0.722] |
| E14_incII ensamble 3 | 0.667 | 0.396 | 0.260 | 0.314 | 0.833 | 0.639 | 0.723 |

## Test fijo (6 pacientes MIT-BIH; entrenado con train MIT-BIH + INCART, early stopping con los 6 de val)

Solo se reporta; no se usó para elegir nada. ~96 % de los SVEB del test son del registro 232.

| modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|
| E12 desplegado (argmax) | 0.748 | 0.196 | 0.827 | 0.316 | 0.991 | 0.972 | 0.981 |
| E12 desplegado (umbral 0.24) | 0.827 | 0.616 | 0.536 | 0.573 | 0.964 | 0.976 | 0.970 |
| E12 (3 semillas) | 0.743 [0.725–0.769] | 0.192 [0.146–0.239] | 0.860 [0.769–0.925] | 0.312 [0.253–0.377] | 0.972 [0.939–0.991] | 0.970 [0.965–0.977] | 0.971 [0.954–0.981] |
| E12 ensamble 3 | 0.746 | 0.185 | 0.918 | 0.308 | 0.991 | 0.975 | 0.983 |
| E13_rrctx (3 semillas) | 0.766 [0.758–0.781] | 0.244 [0.216–0.293] | 0.881 [0.804–0.925] | 0.379 [0.349–0.429] | 0.965 [0.936–0.990] | 0.978 [0.958–0.989] | 0.971 [0.961–0.979] |
| E13_rrctx ensamble 3 | 0.776 | 0.252 | 0.931 | 0.397 | 0.982 | 0.981 | 0.981 |
| E14_incII (3 semillas) | 0.806 [0.776–0.846] | 0.388 [0.278–0.468] | 0.972 [0.952–0.987] | 0.551 [0.431–0.635] | 0.992 [0.991–0.993] | 0.863 [0.692–0.988] | 0.918 [0.816–0.989] |
| E14_incII ensamble 3 | 0.825 | 0.378 | 0.975 | 0.544 | 0.993 | 0.955 | 0.974 |

### Test sin el registro 232 (SVEB n = 61: no permite conclusiones sobre SVEB)

| modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|
| E12 desplegado (argmax) | 0.666 | 0.016 | 0.021 | 0.019 | 0.991 | 0.980 | 0.985 |
| E12 desplegado (umbral 0.24) | 0.651 | 0.098 | 0.009 | 0.016 | 0.964 | 0.983 | 0.974 |
| E12 (3 semillas) | 0.666 [0.650–0.680] | 0.022 [0.000–0.049] | 0.052 [0.000–0.088] | 0.030 [0.000–0.063] | 0.972 [0.939–0.991] | 0.979 [0.977–0.982] | 0.975 [0.957–0.984] |
| E12 ensamble 3 | 0.660 | 0.000 | 0.000 | 0.000 | 0.991 | 0.981 | 0.986 |
| E13_rrctx (3 semillas) | 0.669 [0.660–0.673] | 0.033 [0.033–0.033] | 0.056 [0.020–0.080] | 0.039 [0.025–0.047] | 0.965 [0.936–0.990] | 0.983 [0.968–0.993] | 0.974 [0.962–0.981] |
| E13_rrctx ensamble 3 | 0.682 | 0.049 | 0.111 | 0.068 | 0.982 | 0.987 | 0.984 |
| E14_incII (3 semillas) | 0.647 [0.615–0.671] | 0.022 [0.016–0.033] | 0.131 [0.059–0.250] | 0.037 [0.026–0.058] | 0.992 [0.991–0.993] | 0.865 [0.694–0.988] | 0.919 [0.817–0.990] |
| E14_incII ensamble 3 | 0.665 | 0.016 | 0.083 | 0.027 | 0.993 | 0.956 | 0.974 |

## Lectura

1. **El 0.717 histórico estaba inflado** (registros del mismo paciente en train y evaluación, early stopping sobre el fold evaluado, test dentro de la CV). Con el protocolo corregido E12 da F1-macro 0.637 en CV.
2. **El umbral SVEB 0.24 empeora el modelo** bajo evaluación honesta (PPV SVEB 0.070 vs 0.232 con argmax); el óptimo en OOF es ≈0.50. El test de E12 que se reportaba (0.748) era argmax; con 0.24 el desplegado da SVEB Se 0.616 / PPV 0.536 en test pero PPV 0.009 sin 232.
3. **E13 (contexto RR)**: SVEB PPV +0.153, Se +0.202, 5/5 folds. Costo VEB pequeño que el ensamble compensa.
4. **E14 (INCART derivación II)**: VEB AP 0.742 → 0.905, VEB PPV +0.148 y Se +0.102 en CV (en MIT-BIH: VEB PPV 0.440 → 0.633). SVEB en el pool completo a igual Se mejora (PPV@Se=0.50 0.439 vs 0.384), pero **en MIT-BIH SVEB no mejora**: AP 0.252 vs 0.265 (rangos solapados) y con argmax la PPV baja (0.259 vs 0.347) porque el punto de operación se corre a más Se. Además la partición de folds cambió respecto a E13 (no pareada); desde E15 los folds quedan fijos (`configs/cv_folds_E14.json`).
5. **Test**: E14 ensamble sube F1-macro (0.825) y SVEB Se/PPV en el test completo, pero VEB PPV por semilla varía mucho (0.69–0.99): con 6 pacientes el test no discrimina; la CV es la referencia.
6. **Sin 232 nada detecta los 61 SVEB del test**: la generalización de SVEB a pacientes nuevos sigue siendo el problema abierto → E15 (SVDB, +12 169 SVEB, 66 registros con ≥10 SVEB).

Candidato actual: **E14_incII, ensamble de 3 semillas, argmax** (`models/final/E14_incII/seed{0,1,2}.pt` en Drive). Interfaz sin cambios (latido 250 + 6 RR → 3 clases). En despliegue requiere calcular las 6 features RR de contexto (`src/data/rr_context.py`, 1 latido de retraso) y usar argmax (quitar el umbral 0.24). La derivación II solo afecta a los datos de entrenamiento de INCART.
