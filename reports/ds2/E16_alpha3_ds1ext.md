# DS1 → DS2 (de Chazal) — E16_alpha3_ds1ext

Entrenamiento: MIT-BIH DS1 + fuentes externas del config (`data/processed/mitbih_incartII_svdb`). Umbral SVEB elegido en validación interna de DS1. F mapeado a VEB.

| modelo | regla | conjunto | f1_macro | sveb_se | sveb_ppv | sveb_f1 | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | argmax | DS2 | 0.707 | 0.255 | 0.332 | 0.288 | 0.831 | 0.895 | 0.862 |
| seed0 | argmax | DS2 sin 232 | 0.728 | 0.604 | 0.224 | 0.327 | 0.831 | 0.923 | 0.875 |
| seed0 | t_SVEB=0.35 (DS1) | DS2 | 0.700 | 0.302 | 0.250 | 0.274 | 0.818 | 0.909 | 0.861 |
| seed0 | t_SVEB=0.35 (DS1) | DS2 sin 232 | 0.705 | 0.725 | 0.164 | 0.267 | 0.818 | 0.931 | 0.871 |
| seed1 | argmax | DS2 | 0.708 | 0.249 | 0.355 | 0.293 | 0.787 | 0.949 | 0.861 |
| seed1 | argmax | DS2 sin 232 | 0.729 | 0.593 | 0.242 | 0.344 | 0.787 | 0.952 | 0.862 |
| seed1 | t_SVEB=0.31 (DS1) | DS2 | 0.713 | 0.426 | 0.277 | 0.336 | 0.751 | 0.959 | 0.842 |
| seed1 | t_SVEB=0.31 (DS1) | DS2 sin 232 | 0.688 | 0.785 | 0.148 | 0.248 | 0.751 | 0.961 | 0.843 |
| seed2 | argmax | DS2 | 0.707 | 0.252 | 0.324 | 0.284 | 0.827 | 0.917 | 0.870 |
| seed2 | argmax | DS2 sin 232 | 0.718 | 0.555 | 0.205 | 0.299 | 0.827 | 0.922 | 0.872 |
| seed2 | t_SVEB=0.34 (DS1) | DS2 | 0.708 | 0.343 | 0.262 | 0.297 | 0.806 | 0.926 | 0.862 |
| seed2 | t_SVEB=0.34 (DS1) | DS2 sin 232 | 0.699 | 0.734 | 0.157 | 0.259 | 0.806 | 0.931 | 0.864 |
| ensamble | argmax | DS2 | 0.714 | 0.264 | 0.345 | 0.299 | 0.816 | 0.937 | 0.872 |
| ensamble | argmax | DS2 sin 232 | 0.729 | 0.602 | 0.227 | 0.330 | 0.816 | 0.945 | 0.876 |
| ensamble | t_SVEB=0.37 (DS1) | DS2 | 0.708 | 0.303 | 0.281 | 0.292 | 0.805 | 0.940 | 0.867 |
| ensamble | t_SVEB=0.37 (DS1) | DS2 sin 232 | 0.713 | 0.705 | 0.182 | 0.289 | 0.805 | 0.948 | 0.871 |

Referencia literatura (DS2, SVEB Se/PPV/F1 · VEB Se/PPV): de Chazal 2004 75.9/38.5/0.51 · 77.7/81.9; Llamedo 2011 76.5/41.3/0.53 · 82.9/88.0; Zhang 2014 79.1/36.0/0.495 · 85.5/92.8.
