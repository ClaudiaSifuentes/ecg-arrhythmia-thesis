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

## E15 — + SVDB (78 registros, +12 169 SVEB). Folds pareados con E14 (mapa fijo)

Comparación en los **mismos sujetos y latidos** que E14. Umbral SVEB elegido de forma anidada (en la validación interna de cada fold y semilla; mediana 0.54 → `configs/experiments/E15_svdb.yaml`), igual para ambos modelos.

| vista | modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|---|
| MIT-BIH (37) | E14 umbral anidado | 0.650 [0.641–0.657] | 0.274 [0.258–0.284] | 0.269 [0.251–0.283] | 0.272 [0.255–0.284] | 0.828 [0.823–0.831] | 0.628 [0.614–0.635] | 0.714 [0.706–0.720] |
| MIT-BIH (37) | **E15 umbral anidado** | **0.705** [0.700–0.708] | **0.473** [0.455–0.484] | **0.303** [0.294–0.308] | **0.369** [0.364–0.376] | **0.849** [0.819–0.878] | **0.709** [0.698–0.718] | **0.773** [0.753–0.787] |
| 66 compartidos | E14 umbral anidado | 0.729 [0.702–0.754] | 0.388 [0.294–0.494] | 0.401 [0.355–0.456] | 0.393 [0.322–0.474] | 0.896 [0.893–0.897] | 0.757 [0.746–0.776] | 0.821 [0.815–0.832] |
| 66 compartidos | E15 umbral anidado | 0.751 [0.734–0.769] | 0.572 [0.536–0.608] | 0.341 [0.284–0.383] | 0.427 [0.372–0.470] | 0.906 [0.883–0.925] | 0.801 [0.722–0.854] | 0.849 [0.806–0.888] |
| 66 compartidos | E15 argmax | 0.735 [0.723–0.744] | 0.684 [0.675–0.700] | 0.265 [0.252–0.280] | 0.382 [0.368–0.400] | 0.893 [0.878–0.912] | 0.814 [0.730–0.869] | 0.851 [0.802–0.890] |

Test fijo, ensamble 3 semillas, argmax (el umbral 0.54 se fijó después; su test está pendiente): completo F1-macro 0.822, SVEB Se 0.369 / PPV 0.937, VEB Se 0.984 / PPV 0.973. **Sin 232: SVEB Se 0.098 / PPV 0.158 (n=61)**, el primer modelo que detecta algo de esos SVEB (E14: 0.016 / 0.083).

**Lectura E15.** En MIT-BIH (dominio del test y de la literatura) mejora la PPV y la Se de **ambas** clases. En el pool que incluye INCART cambia PPV de SVEB por Se (−0.06 / +0.18). Con argmax sobrepredice SVEB: α=8 se fijó para una prevalencia de SVEB ~2.6× menor → siguiente experimento E16 (α_SVEB 8 → 3). **Candidato actual: E15 + umbral SVEB 0.54, ensamble de 3 semillas.**

## E16 — focal α_SVEB 8 → 3 (folds pareados con E15)

Con SVDB la prevalencia de SVEB subió ~2.6×; α=8 sobrepredecía SVEB. Con α=3 el modelo queda calibrado: el umbral anidado sale ≈0.46 (≈ argmax), así que **se despliega con argmax, sin umbral**.

| vista | modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 | AP SVEB |
|---|---|---|---|---|---|---|---|---|---|
| MIT-BIH (37) | E15 + umbral anidado (candidato anterior) | 0.705 [0.700–0.708] | 0.473 [0.455–0.484] | 0.303 [0.294–0.308] | 0.369 [0.364–0.376] | 0.849 [0.819–0.878] | 0.709 [0.698–0.718] | 0.773 [0.753–0.787] | 0.306 |
| MIT-BIH (37) | **E16 argmax** | **0.716** [0.710–0.724] | 0.452 [0.433–0.481] | **0.314** [0.291–0.336] | 0.370 [0.349–0.382] | 0.841 [0.822–0.851] | **0.767** [0.729–0.840] | **0.802** [0.775–0.845] | **0.350** |
| 66 compartidos | E15 + umbral anidado | 0.751 [0.734–0.769] | 0.572 [0.536–0.608] | 0.341 [0.284–0.383] | 0.427 [0.372–0.470] | 0.906 [0.883–0.925] | 0.801 [0.722–0.854] | 0.849 [0.806–0.888] | 0.342 |
| 66 compartidos | **E16 argmax** | **0.777** [0.772–0.784] | 0.593 [0.581–0.615] | **0.425** [0.363–0.473] | **0.494** [0.448–0.522] | 0.910 [0.908–0.913] | 0.816 [0.761–0.883] | 0.859 [0.830–0.896] | **0.473** |
| pool completo (144) | E16 ensamble 3, argmax | 0.774 | 0.518 | 0.536 | 0.527 | 0.892 | 0.765 | 0.823 | 0.502 |

Test fijo, ensamble E16 argmax: completo F1-macro 0.774, SVEB Se 0.249 / PPV 0.978, VEB Se 0.983 / PPV 0.963; sin 232 (SVEB n=61): Se 0.033 / PPV 0.222. (E15 con t=0.54 en test: F1-macro 0.784, SVEB Se 0.278 / PPV 0.964; sin 232 Se 0.016 / PPV 0.071.) El test sigue sin discriminar entre modelos.

**Candidato actual: E16 (MIT-BIH + INCART deriv. II + SVDB, RR de contexto, focal γ=2 α=[1,3,3]), ensamble de 3 semillas, argmax.** Siguiente: robustez (E17, augmentation) medida con `scripts/eval_robustness.py`.

## E17 — augmentation en entrenamiento (folds pareados con E16) + robustez

Robustez medida con `scripts/eval_robustness.py`: cada modelo de CV re-evalúa **sus propios pacientes OOF** con la señal degradada (simulación Polar H10: 130 Hz, deriva, movimiento, ruido de contacto escalonado; y error de centrado ±22 ms). Argmax, media de 3 semillas [mín–máx].

| vista | señal | modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|---|---|
| MIT-BIH | limpia | E16 | **0.716** [0.710–0.724] | 0.452 | **0.314** [0.291–0.336] | **0.370** | 0.841 | **0.767** | **0.802** |
| MIT-BIH | limpia | E17 | 0.700 [0.694–0.704] | 0.435 | 0.274 [0.248–0.289] | 0.336 | 0.840 | 0.748 | 0.791 |
| MIT-BIH | Polar sim | E16 | 0.619 [0.586–0.644] | 0.362 | 0.171 [0.081–0.315] | 0.202 | 0.590 | 0.854 | 0.698 |
| MIT-BIH | Polar sim | E17 | **0.672** [0.655–0.687] | **0.421** | **0.214** [0.169–0.261] | **0.283** | **0.763** | 0.764 | **0.763** |
| pool | limpia | E16 | 0.764 [0.757–0.768] | 0.504 | 0.506 | 0.505 | 0.883 | 0.757 | 0.815 |
| pool | limpia | E17 | 0.772 [0.765–0.781] | 0.527 | 0.517 | 0.522 | 0.883 | 0.767 | 0.821 |
| pool | Polar sim | E16 | 0.684 [0.673–0.698] | 0.381 | 0.349 | 0.329 | 0.678 | 0.879 | 0.765 |
| pool | Polar sim | E17 | **0.754** [0.749–0.759] | **0.496** | **0.460** | **0.477** | **0.831** | 0.799 | **0.814** |

El error de centrado (±22 ms) no afecta a ninguno (Δ ≤ 0.01). Test fijo, ensamble: E17 F1-macro 0.789, SVEB Se 0.282 / PPV 0.978 (E16: 0.774, 0.249 / 0.978).

**Lectura E17.** Con señal limpia de MIT-BIH empeora la PPV de SVEB (−0.040, 1/5 folds) → **no se adopta como reemplazo según la regla**. Bajo la degradación simulada tipo Polar H10 es claramente más robusto (F1-macro MIT +0.053; VEB Se +0.17). Si el despliegue es un wearable de pecho, E17 es preferible; la decisión depende del dispositivo. Siguiente propuesta: E18 (augmentation más suave, p=0.25 sin desplazamiento) para buscar ambas cosas.

### Decisión con el destino real del modelo (wearable de pecho)

El modelo se desplegará en un **wearable de pecho**, así que desde E17 la regla de adopción se aplica principalmente a los pacientes MIT-BIH OOF **con la señal degradada**. La guarda con señal limpia es: el F1-macro no baja más de 0.02 y la Se no baja más de 0.03. Con este criterio **E17 pasa a ser el candidato provisional**: con la simulación Polar H10, la PPV de SVEB sube +0.043 y la Se +0.059; la Se de VEB sube +0.173, aunque su PPV baja −0.090. Con la señal limpia el F1-macro baja −0.016, dentro de la guarda. Queda pendiente confirmarlo con **ruido real** de la MIT-BIH Noise Stress Test DB (`src/data/nstdb_noise.py`: em / ma / bw a 12 dB y mezcla a 6 dB), porque la simulación es sintética y no está validada contra un Polar real.
