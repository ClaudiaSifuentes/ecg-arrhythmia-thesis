# Auditoría del pipeline L1 (HeartlyPE) — puntos de mejora priorizados

Rama `data-artifacts`. Cada punto indica la evidencia en el código y si ya está corregido en esta rama.
Prioridad: **P0** = invalida o sesga números reportados; **P1** = afecta reproducibilidad/comparabilidad; **P2** = rendimiento/robustez; **P3** = higiene.

## Limitación del entorno (leer primero)

- Los datos disponibles en el repo son `data/processed/mitbih_aami3/` = **solo MIT-BIH, 100 048 latidos, 44 registros**. El dataset combinado MIT-BIH+INCART (235 566 latidos) está en `.gitignore` (`data/processed/mitbih_incart/X_beats.npy`) y physionet.org está bloqueado por la política de red de esta sesión, así que **no puedo reconstruir INCART ni bajar SVDB/NSTDB**.
- Consecuencia: el baseline oficial que produzco es la **receta E12 re-entrenada solo con MIT-BIH** bajo el protocolo corregido. Sus números **no son comparables** con la tabla original (0.717 F1-macro), que mezcla INCART y otro protocolo. Todas las comparaciones de esta rama son contra ese baseline re-ejecutado.
- Pool de CV = pacientes no-test: 38 pacientes, 86 614 latidos, **solo 1 290 SVEB** (209: 383, 222: 209, 207: 107, …; solo 14 pacientes con ≥10 SVEB). La varianza de SVEB será grande.

---

## P0 — sesgan o invalidan resultados

1. **El test de E12 se reportó con argmax, no con el umbral 0.24.**
   `src/training/eval_report.py:38` usa `logits.argmax`. Re-evaluando `models/checkpoints/E12_best.pt` (`reports/test/E12_best.md`):

   | E12 desplegado, test | F1-macro | SVEB Se | SVEB PPV | VEB Se | VEB PPV |
   |---|---|---|---|---|---|
   | argmax (= números reportados) | 0.748 | 0.196 | 0.827 | 0.991 | 0.972 |
   | umbral SVEB 0.24 | 0.827 | 0.616 | 0.536 | 0.964 | 0.976 |
   | argmax, sin 232 | 0.666 | 0.016 | 0.021 | 0.991 | 0.980 |
   | 0.24, sin 232 (SVEB n=61) | 0.651 | 0.098 | 0.009 | 0.964 | 0.983 |

   La tabla de la tesis mezcla una regla (0.24, "desplegada") con métricas de otra (argmax). Si SageMaker aplica 0.24, el punto de operación real en test es Se 0.62 / PPV 0.54, no Se 0.20 / PPV 0.83. **Corregido:** `src/training/decision.py` es la única regla, leída de `decision:` en el config, y la usan CV, test (`scripts/eval_test.py`) y el barrido.

2. **Early stopping sobre el fold de evaluación en la CV.** `src/training/cross_validate.py:194` pasa `val_loader` (el fold evaluado) a `fit`, que selecciona el checkpoint por `val_loss` de ese fold (`trainer.py:137`). Optimista. **Corregido:** `cv_protocol.inner_split` arma una validación interna (≈20 % de los pacientes de entrenamiento, agrupada por paciente, estratificada, con las 3 clases) solo para early stopping.

3. **La CV incluye a los pacientes de test.** `cross_validate.py:100-116` usa todos los registros de `data_dir`; solo excluye lo que se pase por `--exclude_patients`. Los 6 pacientes de test (incluido 232) entraban en los folds, así que la CV y el test no eran independientes. **Corregido:** `load_dev_data` excluye `TEST_PATIENTS`.

4. **El umbral 0.24 y el modelo se eligieron mirando validación fija + test.** `scripts/threshold.py` ajusta el umbral sobre `VAL_PATIENTS` (6 pacientes; solo 220, 222 y 223 tienen SVEB relevantes); los mismos 6 pacientes hacen early stopping de E01–E12, y el test se miró en cada iteración (`run_experiment.py:205` lo evalúa siempre). El test ya no es un árbitro limpio. **Mitigado:** los umbrales se eligen solo con OOF / validación interna anidada (`scripts/threshold_sweep_oof.py`); el test se reporta, no se usa para decidir.

5. **`cross_validate.py` no puede reproducir E12 tal como está.** Sin `class_weights` ni `criterion`, `fit` ejecuta `torch.tensor(None)` (`trainer.py:87`) → excepción; además siempre aplica SMOTE (`cross_validate.py:162`) y no tiene focal loss. Los números de CV de E12 no salen de este script: no hay código versionado que los reproduzca. **Corregido:** `scripts/run_cv.py` + `configs/experiments/E12.yaml`.

6. **Las exclusiones no se aplican de forma coherente.** `splits.EXCLUDED_RECORDS` incluye 232 y 209 (`splits.py:27`), pero 209 está en `TRAIN_PATIENTS` (`splits.py:64`) y 232 en `TEST_PATIENTS`; `make_patient_wise_splits` no consulta la lista. `build_dataset` sí filtra (`dataset_builder.py:248`), `build_dataset_multi_source` no. Según qué builder generó los `.npy`, 209/232 están o no están. **Pendiente de decisión del autor** (en esta rama: 209 en CV, 232 solo en test).

## P1 — reproducibilidad y comparabilidad

7. **No determinismo.** `set_global_seeds` (`utils/reproducibility.py`) no fija `cudnn.deterministic`, `cudnn.benchmark=False` ni `torch.use_deterministic_algorithms`; en GPU, cuDNN elige kernels no deterministas → F1 0.708/0.724/0.718 con la misma semilla. `run_experiment.py:104,111` fija la semilla 42 sin opción. **Corregido:** `enable_determinism()` + generador explícito para el barajado + número de hilos fijo; `tests/test_protocol.py::test_training_is_bitwise_deterministic` verifica logits idénticos bit a bit entre dos ejecuciones.

8. **La semilla también cambia los folds.** `StratifiedGroupKFold(..., random_state=seed)` (`cross_validate.py:118`): con otra semilla cambian los pacientes de cada fold, así que "3 corridas" mezclan varianza de partición y de entrenamiento, y no se puede comparar fold a fold. **Corregido:** `cv.split_seed` fijo (42); la semilla de entrenamiento solo afecta inicialización y orden.

9. **No se guardaban OOF.** **Corregido:** `reports/oof/oof_{exp}.npz` con `y, prob, logits, fold, patient_id, seeds` + logits de la validación interna (`val_*`), para elegir umbrales/temperatura sin tocar OOF.

10. **Las métricas de CV no incluían PPV/Se por clase ni AUC con umbral.** `compute_metrics` solo devuelve F1-macro, recall/especificidad VEB. **Corregido:** `metrics.class_metrics` (Se/PPV/F1 de SVEB y VEB; PPV=NaN si la clase nunca se predice, para no promediar ceros falsos). Se reportan métricas *pooled* (todas las predicciones OOF) y *fold-mean*, ambas con [min–max] entre semillas.

11. **`ReduceLROnPlateau(patience=7)` con early stopping `patience=8`** (`trainer.py:64`, receta E12): el LR solo puede bajar una vez justo antes de parar → en la práctica no hay scheduling. Hiperparámetro sospechoso; se mantiene en el baseline por fidelidad y es candidato a experimento (p. ej. lr_patience=3).

## P2 — rendimiento / robustez del modelo

12. **Las features RR entran sin escalar** (`fusion.py:50`): RR en segundos (≈0.8), SDNN ≈0.09, RMSSD ≈0.14, ratio ≈1.04 se concatenan con un embedding de 256 dims post-ReLU. Las escalas pequeñas (SDNN/RMSSD) quedan subponderadas al inicio. Candidato: normalización fija (media/DE de train guardadas en config) — cambia el preprocesamiento desplegado, no la interfaz.

13. **Contexto RR pobre para SVEB.** Las 6 features son causales y locales (ventana de 5 RR). Falta RR siguiente (pausa compensatoria) y RR normalizado por el ritmo propio del paciente (la feature más discriminante para SVEB en la literatura, p. ej. de Chazal 2004; Mondéjar-Guerra 2019). `src/data/rr_context.py` implementa un set `context` de 6 columnas (mantiene la forma de la interfaz; exige un latido de retraso en inferencia).

14. **Desajuste de derivación MIT-BIH vs INCART.** `MITBIHDataset.load_data` toma `p_signal[:, 0]` (`mitbih_dataset.py:23`): en MIT-BIH es MLII (salvo el registro 114, donde el canal 0 es V5); en INCART el canal 0 es la derivación **I**, no II. La morfología de INCART que se agrega para SVEB no corresponde a la de MIT-BIH ni, probablemente, a la del dispositivo desplegado.

15. **Latidos centrados en la anotación, no en el pico detectado.** `dataset_builder.py:163` corta la ventana sobre `ann.sample`; en despliegue la ventana se centrará en el pico de Pan–Tompkins/neurokit. El modelo no ve jitter de centrado → justifica augmentation con desplazamiento temporal. (Además, el builder usa `method="neurokit"` (`dataset_builder.py:136`), no Pan–Tompkins como dice la descripción.)

16. **Test de 6 pacientes dominado por 232** (1 380 de 1 441 SVEB). SVEB de test sin 232: n=61. No sirve para estimar generalización de SVEB; la CV pooled de 38 pacientes es la referencia.

17. **Fine-tuning por paciente con partición aleatoria de latidos.** `patient_adaptive.py:68-71` permuta los latidos del registro y toma 20 % para calibrar: latidos vecinos casi idénticos quedan en calibración y evaluación (fuga temporal). Un despliegue real calibra con los primeros N minutos. Baja prioridad (el brief la marca como opcional).

## P3 — higiene

18. `tests/test_splits_patientwise.py` importa `patientwise_split`, que ya no existe; `tests/test_rr_features.py` falla con su propio fixture (picos no ordenados). La suite no corría.
19. `scripts/split_test.py:19` filtra con `np.isin(patient_id, TEST_PATIENTS)` → vacío con ids con prefijo `mit_`; `scripts/threshold.py:27` usa `data_dir` fijo `mitbih_aami3` y aplica SMOTE innecesariamente; `compare_models.py` apunta a `models/E10_best.pt` (ruta inexistente).
20. `configs/*.yaml` no describen E12 (batch 32, lr 1e-3, sin focal ni umbral). **Corregido:** `configs/experiments/E12.yaml` es la receta completa, umbral incluido.
21. `reports/` está entero en `.gitignore`; los entregables (`experiments_log.csv`, OOF, resúmenes) se des-ignoran explícitamente.

## Orden de trabajo aplicado

1. Protocolo (P0 1–5, P1 7–10) → `scripts/run_cv.py`, `src/training/cv_protocol.py`, `decision.py`.
2. Baseline E12 con 3 semillas en el protocolo nuevo → referencia oficial.
3. Barrido de umbrales OOF + estimación anidada.
4. Experimentos de un cambio (ensamble/temperatura y focal α/γ son baratos; contexto RR y augmentation requieren re-entrenar).
