# DS1 → DS2 (de Chazal) — E12_ds1mit

Entrenamiento: solo MIT-BIH DS1 (`data/processed/mitbih_incart`). Umbral SVEB elegido en validación interna de DS1. F mapeado a VEB.

| modelo | regla | conjunto | f1_macro | sveb_se | sveb_ppv | sveb_f1 | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | argmax | DS2 | 0.625 | 0.196 | 0.093 | 0.126 | 0.825 | 0.803 | 0.814 |
| seed0 | argmax | DS2 sin 232 | 0.615 | 0.365 | 0.045 | 0.080 | 0.825 | 0.808 | 0.817 |
| seed0 | t_SVEB=0.63 (DS1) | DS2 | 0.612 | 0.056 | 0.106 | 0.073 | 0.829 | 0.782 | 0.805 |
| seed0 | t_SVEB=0.63 (DS1) | DS2 sin 232 | 0.628 | 0.161 | 0.077 | 0.104 | 0.829 | 0.788 | 0.808 |
| seed1 | argmax | DS2 | 0.613 | 0.177 | 0.093 | 0.122 | 0.840 | 0.731 | 0.782 |
| seed1 | argmax | DS2 sin 232 | 0.613 | 0.418 | 0.056 | 0.099 | 0.840 | 0.745 | 0.789 |
| seed1 | t_SVEB=0.23 (DS1) | DS2 | 0.503 | 0.552 | 0.042 | 0.079 | 0.805 | 0.773 | 0.789 |
| seed1 | t_SVEB=0.23 (DS1) | DS2 sin 232 | 0.490 | 0.765 | 0.015 | 0.029 | 0.805 | 0.786 | 0.796 |
| seed2 | argmax | DS2 | 0.604 | 0.188 | 0.045 | 0.073 | 0.764 | 0.947 | 0.846 |
| seed2 | argmax | DS2 sin 232 | 0.604 | 0.535 | 0.032 | 0.060 | 0.764 | 0.950 | 0.847 |
| seed2 | t_SVEB=0.65 (DS1) | DS2 | 0.629 | 0.084 | 0.091 | 0.087 | 0.772 | 0.931 | 0.844 |
| seed2 | t_SVEB=0.65 (DS1) | DS2 sin 232 | 0.648 | 0.304 | 0.081 | 0.128 | 0.772 | 0.933 | 0.845 |
| ensamble | argmax | DS2 | 0.626 | 0.180 | 0.073 | 0.104 | 0.814 | 0.878 | 0.845 |
| ensamble | argmax | DS2 sin 232 | 0.624 | 0.443 | 0.046 | 0.083 | 0.814 | 0.883 | 0.847 |
| ensamble | t_SVEB=0.60 (DS1) | DS2 | 0.624 | 0.072 | 0.094 | 0.082 | 0.821 | 0.846 | 0.833 |
| ensamble | t_SVEB=0.60 (DS1) | DS2 sin 232 | 0.642 | 0.239 | 0.078 | 0.118 | 0.821 | 0.851 | 0.836 |

Referencia literatura (DS2, SVEB Se/PPV/F1 · VEB Se/PPV): de Chazal 2004 75.9/38.5/0.51 · 77.7/81.9; Llamedo 2011 76.5/41.3/0.53 · 82.9/88.0; Zhang 2014 79.1/36.0/0.495 · 85.5/92.8.
