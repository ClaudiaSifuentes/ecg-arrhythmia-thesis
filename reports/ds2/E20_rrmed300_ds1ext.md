# DS1 → DS2 (de Chazal) — E20_rrmed300_ds1ext

Entrenamiento: MIT-BIH DS1 + fuentes externas del config (`data/processed/mitbih_incartII_svdb`). Umbral SVEB elegido en validación interna de DS1. F mapeado a VEB.

| modelo | regla | conjunto | f1_macro | sveb_se | sveb_ppv | sveb_f1 | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | argmax | DS2 | 0.708 | 0.252 | 0.328 | 0.285 | 0.847 | 0.897 | 0.871 |
| seed0 | argmax | DS2 sin 232 | 0.733 | 0.633 | 0.231 | 0.338 | 0.847 | 0.911 | 0.878 |
| seed0 | t_SVEB=0.38 (DS1) | DS2 | 0.703 | 0.297 | 0.265 | 0.280 | 0.831 | 0.900 | 0.864 |
| seed0 | t_SVEB=0.38 (DS1) | DS2 sin 232 | 0.712 | 0.729 | 0.179 | 0.287 | 0.831 | 0.915 | 0.871 |
| seed1 | argmax | DS2 | 0.718 | 0.298 | 0.388 | 0.337 | 0.838 | 0.858 | 0.848 |
| seed1 | argmax | DS2 sin 232 | 0.741 | 0.680 | 0.262 | 0.378 | 0.838 | 0.891 | 0.864 |
| seed1 | t_SVEB=0.53 (DS1) | DS2 | 0.682 | 0.177 | 0.356 | 0.236 | 0.842 | 0.836 | 0.839 |
| seed1 | t_SVEB=0.53 (DS1) | DS2 sin 232 | 0.750 | 0.588 | 0.311 | 0.406 | 0.842 | 0.880 | 0.861 |
| seed2 | argmax | DS2 | 0.697 | 0.219 | 0.286 | 0.248 | 0.830 | 0.926 | 0.875 |
| seed2 | argmax | DS2 sin 232 | 0.722 | 0.591 | 0.209 | 0.309 | 0.830 | 0.929 | 0.876 |
| seed2 | t_SVEB=0.35 (DS1) | DS2 | 0.695 | 0.294 | 0.223 | 0.254 | 0.814 | 0.932 | 0.869 |
| seed2 | t_SVEB=0.35 (DS1) | DS2 sin 232 | 0.698 | 0.740 | 0.150 | 0.250 | 0.814 | 0.934 | 0.870 |
| ensamble | argmax | DS2 | 0.715 | 0.263 | 0.351 | 0.301 | 0.839 | 0.913 | 0.875 |
| ensamble | argmax | DS2 sin 232 | 0.741 | 0.658 | 0.249 | 0.362 | 0.839 | 0.923 | 0.879 |
| ensamble | t_SVEB=0.41 (DS1) | DS2 | 0.706 | 0.264 | 0.299 | 0.281 | 0.837 | 0.907 | 0.871 |
| ensamble | t_SVEB=0.41 (DS1) | DS2 sin 232 | 0.730 | 0.709 | 0.219 | 0.335 | 0.837 | 0.919 | 0.876 |

Referencia literatura (DS2, SVEB Se/PPV/F1 · VEB Se/PPV): de Chazal 2004 75.9/38.5/0.51 · 77.7/81.9; Llamedo 2011 76.5/41.3/0.53 · 82.9/88.0; Zhang 2014 79.1/36.0/0.495 · 85.5/92.8.
