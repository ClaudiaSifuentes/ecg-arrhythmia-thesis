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

## Robustez con ruido REAL (NSTDB) — E16 vs E17 vs E18

Ensamble de 3 semillas (lo que se despliega), pacientes MIT-BIH OOF, argmax. em = artefacto de electrodo, ma = muscular, bw = deriva de línea base (NSTDB, ruido real grabado en voluntarios); "all6" = mezcla al azar a 6 dB.

| señal | E16 F1m | E16 SVEB PPV / Se | E16 VEB PPV / Se | E17 F1m | E17 SVEB PPV / Se | E17 VEB PPV / Se | E18 F1m | E18 SVEB PPV / Se | E18 VEB PPV / Se |
|---|---|---|---|---|---|---|---|---|---|
| limpia | **0.725** | **0.331** / 0.453 | 0.779 / 0.855 | 0.706 | 0.281 / 0.436 | 0.762 / 0.844 | 0.712 | 0.321 / 0.444 | 0.733 / 0.859 |
| Polar sim (sintética) | 0.671 | 0.282 / 0.354 | 0.907 / 0.603 | 0.693 | 0.251 / 0.434 | 0.796 / 0.776 | **0.694** | 0.265 / 0.429 | 0.776 / 0.789 |
| NSTDB em 12 dB | **0.701** | **0.305** / 0.447 | 0.695 / 0.861 | 0.681 | 0.250 / 0.418 | 0.690 / 0.849 | 0.685 | 0.282 / 0.436 | 0.657 / 0.864 |
| NSTDB ma 12 dB | **0.652** | 0.281 / 0.298 | 0.792 / 0.621 | 0.623 | 0.244 / 0.287 | 0.745 / 0.559 | 0.638 | 0.282 / 0.300 | 0.714 / 0.610 |
| NSTDB bw 12 dB | **0.700** | **0.276** / 0.436 | 0.772 / 0.806 | 0.683 | 0.247 / 0.417 | 0.747 / 0.788 | 0.682 | 0.255 / 0.411 | 0.717 / 0.812 |
| NSTDB mezcla 6 dB | **0.601** | **0.279** / 0.298 | 0.574 / 0.544 | 0.585 | 0.229 / 0.278 | 0.565 / 0.535 | 0.593 | 0.249 / 0.285 | 0.557 / 0.557 |

**Lectura.** La augmentation sintética (E17, E18) solo ayuda con el simulador sintético, cuyo ruido se parece al de la propia augmentation. **Con ruido real no ayuda: E16 queda igual o mejor en todas las condiciones.** Se revierte la adopción provisional de E17 y **E16 sigue siendo el candidato**. Para la tesis: evaluar la robustez solo con ruido simulado puede llevar a conclusiones equivocadas.

**Cambio de protocolo:** desde aquí cada registro de NSTDB se divide en el tiempo. La 1.ª mitad queda para entrenamiento (E19) y la 2.ª mitad solo para evaluación. Las robusteces de E16, E17 y E18 se recalculan con la 2.ª mitad (sección 18 del notebook) para que la comparación con E19 sea limpia.

## E19 — augmentation con ruido REAL (NSTDB 1.ª mitad), evaluado con la 2.ª mitad

| vista MIT, 3 semillas | señal | E16 | E19 |
|---|---|---|---|
| F1-macro | limpia | **0.716** [0.710–0.724] | 0.688 [0.672–0.709] |
| SVEB PPV | limpia | **0.314** [0.291–0.336] | 0.250 [0.194–0.279] (0/5 folds mejor) |
| F1-macro | músculo 12 dB | 0.632 | **0.674** |
| SVEB PPV / Se | músculo 12 dB | 0.252 / 0.297 | 0.235 / **0.418** |
| F1-macro | mezcla 6 dB | 0.582 | **0.637** |
| VEB PPV / Se | mezcla 6 dB | 0.539 / 0.562 | **0.596 / 0.790** |
| SVEB PPV | electrodo / deriva 12 dB | **0.275 / 0.257** | 0.242 / 0.232 |

**Decisión: no adoptar.** Con señal limpia, E19 baja el F1-macro 0.028, por encima del margen de 0.02, y baja la PPV de SVEB en todos los folds. Con ruido real gana sensibilidad cuando el ruido es fuerte, pero la PPV de SVEB baja en 3 de las 4 condiciones.

# Cierre de la etapa de iteración
**Modelo final: E16, ensamble de 3 semillas, argmax** → `reports/MODEL_CARD_E16.md`. Se probaron los ítems del plan en este orden: protocolo, umbral, contexto RR, más datos de SVEB (INCART con derivación II y SVDB), α de la focal loss, ensamble y calibración, y augmentation sintética y real. Quedan sin probar el label smoothing y los ajustes de arquitectura. Según el plan, estos van solo si lo anterior se agota.

# Comparación con la literatura — protocolo DS1 → DS2 (de Chazal)

Entrenado con DS1 (22 registros MIT-BIH, más INCART II y SVDB en la variante "ext"), con el umbral SVEB elegido en una validación interna de DS1, y **evaluado en DS2 (22 registros) sin haberlo mirado**. Ensamble de 3 semillas, argmax. El proyecto agrupa F con VEB (los papers lo dejan aparte).

| método | VEB Se / PPV | SVEB Se / PPV | SVEB F1 |
|---|---|---|---|
| de Chazal 2004 | 77.7 / 81.9 | 75.9 / 38.5 | 0.51 |
| Llamedo 2011 | 82.9 / 88.0 | 76.5 / 41.3 | 0.53 |
| Zhang 2014 | 85.5 / 92.8 | 79.1 / 36.0 | 0.495 |
| E12 (solo DS1 MIT-BIH) | 81.4 / 87.8 | 18.0 / 7.3 | 0.10 |
| E16 (solo DS1 MIT-BIH) | 76.8 / 91.8 | 5.1 / 14.6 | 0.08 |
| E16 (DS1 + INCART II + SVDB) | 81.6 / **93.7** | 26.4 / 34.5 | 0.30 |
| **E20 (DS1 + INCART II + SVDB)** | **83.9** / 91.3 | 26.3 / 35.1 | 0.30 |
| *E20 ext, DS2 sin el registro 232* | 83.9 / 92.3 | **65.8** / 24.9 | 0.36 |

**Lectura.** En VEB el modelo está a la altura de la literatura, con mejor PPV que de Chazal y Llamedo. En SVEB queda muy por debajo, y casi toda la diferencia viene de **un solo registro, el 232**: tiene ≈1 380 de los ≈1 830 SVEB de DS2, y el modelo detecta pocos de ellos. Sin el 232, la Se de SVEB sube a 0.66 con PPV 0.25. En el 232 los SVEB son el ritmo **mayoritario** (≈78 % de sus latidos), así que cualquier referencia "del ritmo propio del paciente" (la media de 50 o la mediana de 300) se calcula sobre los mismos SVEB y no los marca como prematuros. Entrenado solo con DS1, la CNN generaliza muy mal en SVEB (Se 0.05): con tan pocos pacientes con SVEB, la morfología no alcanza. Los métodos clásicos con RR y un clasificador lineal generalizan mejor en esa condición.

## E20 — RR / mediana de 300 latidos (folds pareados con E16)

| vista MIT, 3 semillas | E16 | **E20** |
|---|---|---|
| F1-macro | 0.716 [0.710–0.724] | **0.727** [0.713–0.735] |
| SVEB Se | 0.452 [0.433–0.481] | **0.487** [0.457–0.540] |
| SVEB PPV | 0.314 [0.291–0.336] | **0.360** [0.350–0.372] (4/5 folds) |
| SVEB F1 | 0.370 | **0.414** |
| AP SVEB | 0.350 | **0.415** |
| VEB Se / PPV | 0.841 / 0.767 | 0.845 / 0.748 |
| Ensamble con ruido real (em 12 dB): F1m, SVEB PPV | 0.698, 0.301 | **0.714, 0.360** |

**Se adopta E20.** Advertencia: en el test fijo (que incluye el 232), la Se de SVEB del ensamble baja de 0.249 a 0.130 por el problema descrito arriba. En DS2 queda igual que E16. El test no se usa para decidir, pero este modo de falla se documenta.

## E21 — híbrido CNN (E20) + modelo solo-RR (folds y validación interna de E20)

Un solo cambio sobre E20: se promedia la probabilidad de la CNN con la de un modelo que ve **solo las 6 entradas RR** (las mismas que ya recibe la CNN, sin señal nueva): p = (1 − w)·p_CNN + w·p_RR. El modelo RR se entrena en cada fold con exactamente los latidos con los que se entrenó la CNN. El peso w se elige por (semilla, fold) en la validación interna (mediana 0.25 con gradient boosting). Los umbrales también salen de la validación interna: t_SVEB 0.40 y t_VEB 0.47 son la mediana de las 15 elecciones. E20 usa los suyos, 0.43 / 0.60, elegidos igual. No se miró ningún paciente OOF para elegir nada. Script: `scripts/hybrid_rr.py`; OOF: `reports/oof/oof_E21_hybrid_gbm.npz`.

| vista MIT (37 sujetos), 3 semillas | E20 argmax | E20 umbrales val (0.43/0.60) | **E21 gbm, umbrales val (0.40/0.47)** |
|---|---|---|---|
| F1-macro | 0.727 [0.713–0.735] | 0.726 [0.721–0.732] | **0.745** [0.739–0.753] |
| SVEB Se | 0.487 [0.457–0.540] | **0.627** [0.568–0.676] | 0.604 [0.574–0.643] |
| SVEB PPV | 0.360 [0.350–0.372] | 0.307 [0.294–0.322] | **0.367** [0.363–0.370] |
| SVEB F1 | 0.414 [0.396–0.440] | 0.412 [0.388–0.436] | **0.456** [0.448–0.464] |
| VEB Se | **0.845** [0.835–0.858] | 0.741 [0.721–0.756] | 0.796 [0.772–0.809] |
| VEB PPV | 0.748 [0.700–0.801] | **0.848** [0.814–0.915] | 0.812 [0.788–0.857] |
| AP SVEB / PPV con Se = 0.5 | 0.415 / 0.394 | (igual) | **0.422 / 0.426** |
| AP VEB / PPV con Se = 0.85 | 0.844 / 0.739 | (igual) | **0.852 / 0.763** |

Comparación fold a fold (E21 contra E20, ambos con sus umbrales de validación): SVEB PPV +0.060 (3/5 folds), F1 +0.044, Se −0.023. VEB Se +0.055 (4/5), PPV −0.036 (0/5, aviso). **Cumple la regla en la vista MIT.** Con umbrales anidados por fold (`nested_both`) también la cumple: SVEB PPV +0.057 (4/5), Se −0.020, VEB Se +0.042. Si ambos usan argmax, no la cumple: la Se de VEB baja 0.045.

Otras vistas (E21 contra E20, ambos con sus umbrales de validación):

| vista | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | regla |
|---|---|---|---|---|---|---|
| sin SVDB (66 sujetos) | −0.012 | **+0.059** (3/5) | +0.046 | +0.057 | −0.014 | cumple |
| pool completo (144) | **−0.069** (1/5) | +0.032 | −0.013 | +0.061 | −0.036 | **no cumple** |

Con regresión logística en lugar de gradient boosting no sirve: AP SVEB 0.415 → 0.384 y Se de VEB −0.046 → descartado.

**Lectura.** El modelo RR mejora el *ranking* (sube el AP y la PPV a Se fija en las dos clases) y, con los umbrales elegidos en validación, la **F1 de SVEB pasa de 0.41 a 0.46 en MIT-BIH**. En los sujetos de SVDB, en cambio, baja la Se de SVEB. SVDB es Holter ambulatorio y se parece al uso del wearable, así que esa baja es un costo real. **Decisión inicial: candidato condicional.** Se adopta solo si en Colab (sección 20) pasa el criterio wearable con ruido real NSTDB (guarda de −0.02 en F1-macro y −0.03 en Se) y si en DS1→DS2 no empeora a E20. Si no, se mantiene E20.

### E21 — resultado de Colab (sección 20) y decisión final

**Robustez con ruido real (criterio wearable; vista MIT, ensamble de 3 semillas, OOF).** La fila "limpio" coincide con la CV local (0.745 / SVEB F1 0.456 por semilla). Esto confirma que el híbrido de Colab es el mismo.

| perturbación | F1-macro E20 → E21 | SVEB F1 E20 → E21 | VEB F1 E20 → E21 |
|---|---|---|---|
| limpio | 0.741 → **0.760** | 0.431 → **0.480** | 0.815 → **0.821** |
| electrodo 12 dB | 0.714 → **0.736** | 0.405 → **0.438** | 0.765 → **0.794** |
| músculo 12 dB | 0.658 → **0.667** | 0.300 → **0.362** | **0.701** → 0.668 |
| deriva 12 dB | 0.710 → **0.722** | 0.376 → **0.419** | 0.778 → 0.774 |
| mezcla 6 dB | 0.603 → **0.646** | 0.279 → **0.337** | 0.573 → **0.633** |
| Polar H10 simulado | **0.680** → 0.673 | 0.338 → **0.364** | **0.727** → 0.684 |

→ **Cumple el criterio wearable.** Con ruido real, E21 es igual o mejor que E20 en F1-macro y en SVEB F1, salvo en Polar simulado (−0.007).

**DS1 → DS2 (literatura), ensamble, w y umbral elegidos en la validación interna de DS1:**

| entrenamiento | modelo | SVEB Se / PPV / F1 | SVEB sin 232 Se / PPV | VEB Se / PPV |
|---|---|---|---|---|
| DS1 + INCART + SVDB | E20 (argmax) | **26.3 / 35.1 / 0.30** | **65.8 / 24.9** | **83.9 / 91.3** |
| DS1 + INCART + SVDB | E21 (w = 0.50, argmax) | 11.7 / 20.2 / 0.15 | 47.2 / 20.0 | 80.5 / 83.9 |
| DS1 + INCART + SVDB | solo RR-gbm | 6.5 / 8.3 / 0.07 | 26.6 / 8.3 | 65.7 / 64.0 |
| solo DS1 MIT-BIH | E20 (argmax) | 5.0 / 16.2 / 0.08 | 19.2 / 15.4 | 77.5 / 89.5 |
| solo DS1 MIT-BIH | E21 (w = 0.90, argmax) | 4.1 / 11.4 / 0.06 | 16.3 / 11.2 | 72.8 / 57.2 |

→ **Empeora a E20 en DS2 en todas las métricas.** El modelo solo-RR no generaliza a los pacientes de DS2: su SVEB F1 es 0.07. La validación interna de DS1 tiene pocos sujetos y le dio a ese modelo un peso alto (w de 0.50 a 0.90).

**Test fijo** (solo se reporta; ensamble con los umbrales de la CV):
- **Completo:** SVEB Se 0.130 → 0.033 (otra vez el 232) y VEB PPV 0.889 → 0.975.
- **Sin el 232** (61 SVEB): SVEB Se 0.033 → 0.098 y PPV 0.200 → 0.429.

**Decisión final: E21 no se adopta y se mantiene E20.** Se había fijado de antemano la condición "en DS1→DS2 no empeora a E20", y no se cumple. La mejora en la CV y con ruido es real para pacientes cuya distribución de RR se parece a la del entrenamiento. Pero un modelo que solo ve las RR no transfiere a pacientes nuevos de otra base. Es justo lo contrario de lo que se esperaba de los métodos clásicos de RR, que además usan morfología. El código (`--hybrid`) y el modelo RR (`models/checkpoints/E21_rr_gbm.joblib`) quedan para reproducir.
