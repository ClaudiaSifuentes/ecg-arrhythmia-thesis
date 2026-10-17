# Ficha del modelo candidato — HeartlyPE L1 · E16 (ensamble de 3 semillas)

Reemplaza a E12 **sin cambiar la interfaz**: latido de 250 muestras + 6 entradas RR → 3 clases (N, SVEB, VEB).
Código de inferencia: `src/inference/heartly_e16.py` (`HeartlyE16`, `rr_features_from_peaks`).

## Receta (cambios acumulados respecto de E12, cada uno adoptado con su experimento: `reports/experiments_log.csv`)
| | E12 | E16 |
|---|---|---|
| Datos de entrenamiento | MIT-BIH + INCART (derivación **I**) | MIT-BIH + INCART (derivación **II**) + **SVDB** (78 registros, +12 169 SVEB) |
| 6 entradas RR | RR, RR_prev, media5, SDNN, RMSSD, RR/RR_prev | **contexto**: RR, RR_next, media5, RR/media50, RR_next/RR, RR/RR_prev |
| Focal loss | γ=2, α=[1, 8, 3] | γ=2, α=[1, **3**, 3] |
| Regla de decisión | argmax + umbral SVEB 0.24 | **argmax** (umbral elegido en OOF ≈ 0.46–0.50) |
| Modelo | 1 red | promedio de 3 redes (semillas 0, 1, 2) |
| Arquitectura, optimizador, épocas | FusionClassifier 158 595 parámetros, Adam 5e-4, wd 1e-4, batch 256, máx. 50, patience 8 | **sin cambios** |

## Validación cruzada — pacientes MIT-BIH (los mismos 38 registros / 37 sujetos y los mismos latidos en ambos)
StratifiedGroupKFold k=5 por sujeto real, early stopping con validación interna por sujeto, 3 semillas (media [mín–máx]).

| modelo | f1_macro | sveb_se | sveb_ppv | sveb_f1 | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|
| E12 receta, regla desplegada (t_SVEB=0.24), 3 semillas | 0.523 [0.491–0.553] | 0.543 [0.466–0.591] | 0.063 [0.032–0.089] | 0.112 [0.059–0.155] | 0.845 [0.809–0.875] | 0.459 [0.392–0.513] | 0.593 [0.537–0.647] |
| E12 receta, argmax, 3 semillas | 0.587 [0.546–0.652] | 0.212 [0.115–0.326] | 0.241 [0.083–0.403] | 0.224 [0.096–0.360] | 0.859 [0.826–0.886] | 0.457 [0.392–0.504] | 0.595 [0.540–0.643] |
| E12 receta, ensamble 3, argmax | 0.587 | 0.195 | 0.220 | 0.207 | 0.868 | 0.468 | 0.608 |
| **E16, 3 semillas, argmax** | 0.716 [0.710–0.724] | 0.452 [0.433–0.481] | 0.314 [0.291–0.336] | 0.370 [0.349–0.382] | 0.841 [0.822–0.851] | 0.767 [0.729–0.840] | 0.802 [0.775–0.845] |
| **E16 ensamble 3, argmax (candidato)** | 0.725 | 0.453 | 0.331 | 0.382 | 0.855 | 0.779 | 0.815 |

## Test fijo (6 pacientes MIT-BIH; ~96 % de los SVEB del test son del registro 232)
| modelo | F1-macro | SVEB Se | SVEB PPV | SVEB F1 | VEB Se | VEB PPV | VEB F1 |
|---|---|---|---|---|---|---|---|
| E12 desplegado, umbral 0.24 | 0.827 | 0.616 | 0.536 | 0.573 | 0.964 | 0.976 | 0.970 |
| E12 desplegado, argmax (lo que se reportaba) | 0.748 | 0.196 | 0.827 | 0.316 | 0.991 | 0.972 | 0.981 |
| E12 receta reentrenada, ensamble 3 | 0.746 | 0.185 | 0.918 | 0.308 | 0.991 | 0.975 | 0.983 |
| **E16 ensamble 3 (candidato)** | 0.774 | 0.249 | **0.978** | 0.397 | 0.983 | 0.963 | 0.973 |
| *Sin 232 (SVEB n=61):* E12 desplegado 0.24 | 0.651 | 0.098 | 0.009 | 0.016 | 0.964 | 0.983 | 0.974 |
| *Sin 232:* **E16 ensamble 3** | **0.679** | 0.033 | **0.222** | 0.057 | 0.983 | 0.987 | 0.985 |

El test es pequeño y se miró durante el desarrollo de E01–E12: **no discrimina entre modelos**. La referencia es la CV.

## Robustez (pacientes MIT-BIH OOF, ensamble; ruido real NSTDB = 2.ª mitad de cada registro)
| señal | F1-macro | SVEB PPV / Se | VEB PPV / Se |
|---|---|---|---|
| limpia | 0.725 | 0.331 / 0.453 | 0.779 / 0.855 |
| NSTDB electrodo 12 dB | 0.698 | 0.301 / 0.432 | 0.692 / 0.864 |
| NSTDB músculo 12 dB | 0.655 | 0.265 / 0.300 | 0.789 / 0.648 |
| NSTDB deriva 12 dB | 0.699 | 0.273 / 0.435 | 0.768 / 0.811 |
| NSTDB mezcla 6 dB | 0.599 | 0.247 / 0.287 | 0.579 / 0.570 |

Con ruido muscular y con mezcla fuerte cae la Se de VEB (≈0.57–0.65). La augmentation (sintética en E17/E18, con ruido real en E19) no mejoró la PPV sin costo en señal limpia, así que no se adoptó (ver RESULTS.md). Para un wearable de pecho se recomienda **un índice de calidad de señal** que marque como "no legibles" los tramos muy ruidosos en vez de clasificarlos.

## Requisitos para desplegar en SageMaker
1. Calcular las 6 RR de contexto con `rr_features_from_peaks` (mismo código que en entrenamiento). Exige **1 latido de retraso** (RR_next).
2. **Quitar el umbral SVEB 0.24**: la regla es argmax del promedio de las 3 redes.
3. Cargar los 3 pesos `models/checkpoints/E16_final_seed{0,1,2}.pt` (entrenados con train MIT-BIH + INCART II + SVDB, early stopping con los 6 pacientes de val).
4. Preprocesamiento del latido sin cambios: pasabanda 0.5–40 Hz, z-score del registro, ventana [R-100, R+150) a 360 Hz.

## Limitaciones
- SVEB sigue siendo la clase débil (PPV ≈ 0.33 en MIT-BIH CV); casi no detecta los SVEB del test fuera del registro 232.
- La robustez se midió con ruido añadido a latidos ya segmentados: las RR siguen limpias (en un wearable real el ruido también afecta la detección de picos R), así que la degradación real puede ser mayor.
- No hay validación con datos reales del dispositivo (Polar H10 u otro). Es la validación pendiente más importante.
- F (fusión) se agrupa con VEB (decisión del proyecto, distinta de AAMI EC57).

## Verificación de los pesos
`HeartlyE16([models/checkpoints/E16_final_seed0.pt, …seed1.pt, …seed2.pt])` sobre el test fijo reproduce exactamente `reports/test/E16_alpha3.md`: F1-macro 0.774, SVEB Se 0.249 / PPV 0.978, VEB Se 0.983 / PPV 0.963. Sin 232: F1-macro 0.679, SVEB Se 0.033 / PPV 0.222. Epoch elegida por early stopping en `models/checkpoints/E16_final_seed*.json`.
