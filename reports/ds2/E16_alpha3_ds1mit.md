# DS1 → DS2 (de Chazal) — E16_alpha3_ds1mit

Entrenamiento: solo MIT-BIH DS1 (`data/processed/mitbih_incartII_svdb`). Umbral SVEB elegido en validación interna de DS1. F mapeado a VEB.

| modelo | regla | conjunto | f1_macro | sveb_se | sveb_ppv | sveb_f1 | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | argmax | DS2 | 0.631 | 0.113 | 0.135 | 0.123 | 0.781 | 0.848 | 0.813 |
| seed0 | argmax | DS2 sin 232 | 0.643 | 0.306 | 0.094 | 0.144 | 0.781 | 0.851 | 0.814 |
| seed0 | t_SVEB=0.65 (DS1) | DS2 | 0.614 | 0.038 | 0.175 | 0.063 | 0.784 | 0.845 | 0.814 |
| seed0 | t_SVEB=0.65 (DS1) | DS2 sin 232 | 0.638 | 0.112 | 0.131 | 0.121 | 0.784 | 0.849 | 0.815 |
| seed1 | argmax | DS2 | 0.612 | 0.033 | 0.124 | 0.052 | 0.750 | 0.907 | 0.821 |
| seed1 | argmax | DS2 sin 232 | 0.642 | 0.132 | 0.122 | 0.127 | 0.750 | 0.910 | 0.822 |
| seed1 | t_SVEB=0.07 (DS1) | DS2 | 0.346 | 0.585 | 0.026 | 0.049 | 0.739 | 0.925 | 0.822 |
| seed1 | t_SVEB=0.07 (DS1) | DS2 sin 232 | 0.333 | 0.957 | 0.010 | 0.021 | 0.739 | 0.928 | 0.823 |
| seed2 | argmax | DS2 | 0.625 | 0.053 | 0.148 | 0.078 | 0.758 | 0.921 | 0.831 |
| seed2 | argmax | DS2 sin 232 | 0.662 | 0.208 | 0.144 | 0.170 | 0.758 | 0.936 | 0.837 |
| seed2 | t_SVEB=0.47 (DS1) | DS2 | 0.623 | 0.050 | 0.139 | 0.074 | 0.759 | 0.919 | 0.831 |
| seed2 | t_SVEB=0.47 (DS1) | DS2 sin 232 | 0.659 | 0.199 | 0.135 | 0.161 | 0.759 | 0.934 | 0.837 |
| ensamble | argmax | DS2 | 0.625 | 0.051 | 0.146 | 0.076 | 0.768 | 0.918 | 0.836 |
| ensamble | argmax | DS2 sin 232 | 0.659 | 0.195 | 0.137 | 0.161 | 0.768 | 0.922 | 0.838 |
| ensamble | t_SVEB=0.48 (DS1) | DS2 | 0.622 | 0.044 | 0.126 | 0.065 | 0.770 | 0.917 | 0.837 |
| ensamble | t_SVEB=0.48 (DS1) | DS2 sin 232 | 0.653 | 0.172 | 0.121 | 0.142 | 0.770 | 0.921 | 0.839 |

Referencia literatura (DS2, SVEB Se/PPV/F1 · VEB Se/PPV): de Chazal 2004 75.9/38.5/0.51 · 77.7/81.9; Llamedo 2011 76.5/41.3/0.53 · 82.9/88.0; Zhang 2014 79.1/36.0/0.495 · 85.5/92.8.
