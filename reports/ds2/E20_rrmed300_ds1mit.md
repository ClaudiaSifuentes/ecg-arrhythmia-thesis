# DS1 → DS2 (de Chazal) — E20_rrmed300_ds1mit

Entrenamiento: solo MIT-BIH DS1 (`data/processed/mitbih_incartII_svdb`). Umbral SVEB elegido en validación interna de DS1. F mapeado a VEB.

| modelo | regla | conjunto | f1_macro | sveb_se | sveb_ppv | sveb_f1 | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | argmax | DS2 | 0.605 | 0.068 | 0.170 | 0.097 | 0.823 | 0.708 | 0.761 |
| seed0 | argmax | DS2 sin 232 | 0.632 | 0.206 | 0.132 | 0.161 | 0.823 | 0.712 | 0.763 |
| seed0 | t_SVEB=0.57 (DS1) | DS2 | 0.596 | 0.041 | 0.186 | 0.067 | 0.824 | 0.707 | 0.761 |
| seed0 | t_SVEB=0.57 (DS1) | DS2 sin 232 | 0.628 | 0.136 | 0.159 | 0.147 | 0.824 | 0.710 | 0.763 |
| seed1 | argmax | DS2 | 0.606 | 0.043 | 0.153 | 0.067 | 0.679 | 0.944 | 0.790 |
| seed1 | argmax | DS2 sin 232 | 0.643 | 0.174 | 0.152 | 0.162 | 0.679 | 0.946 | 0.790 |
| seed1 | t_SVEB=0.43 (DS1) | DS2 | 0.610 | 0.058 | 0.130 | 0.080 | 0.679 | 0.944 | 0.790 |
| seed1 | t_SVEB=0.43 (DS1) | DS2 sin 232 | 0.643 | 0.233 | 0.127 | 0.165 | 0.679 | 0.946 | 0.791 |
| seed2 | argmax | DS2 | 0.631 | 0.071 | 0.151 | 0.096 | 0.783 | 0.886 | 0.831 |
| seed2 | argmax | DS2 sin 232 | 0.669 | 0.275 | 0.145 | 0.190 | 0.783 | 0.907 | 0.840 |
| seed2 | t_SVEB=0.38 (DS1) | DS2 | 0.628 | 0.088 | 0.103 | 0.095 | 0.782 | 0.887 | 0.832 |
| seed2 | t_SVEB=0.38 (DS1) | DS2 sin 232 | 0.656 | 0.351 | 0.101 | 0.157 | 0.782 | 0.908 | 0.840 |
| ensamble | argmax | DS2 | 0.624 | 0.050 | 0.162 | 0.076 | 0.775 | 0.895 | 0.831 |
| ensamble | argmax | DS2 sin 232 | 0.661 | 0.192 | 0.154 | 0.171 | 0.775 | 0.899 | 0.832 |
| ensamble | t_SVEB=0.48 (DS1) | DS2 | 0.620 | 0.042 | 0.139 | 0.065 | 0.776 | 0.895 | 0.831 |
| ensamble | t_SVEB=0.48 (DS1) | DS2 sin 232 | 0.653 | 0.166 | 0.135 | 0.148 | 0.776 | 0.899 | 0.833 |

Referencia literatura (DS2, SVEB Se/PPV/F1 · VEB Se/PPV): de Chazal 2004 75.9/38.5/0.51 · 77.7/81.9; Llamedo 2011 76.5/41.3/0.53 · 82.9/88.0; Zhang 2014 79.1/36.0/0.495 · 85.5/92.8.
