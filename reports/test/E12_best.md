Decision rule (config): {'t_sveb': 0.24, 't_veb': None}

| model | rule | subset | f1_macro | auc_macro | sveb_n | sveb_se | sveb_ppv | sveb_f1 | veb_n | veb_se | veb_ppv | veb_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E12_best | argmax | full | 0.748 | 0.941 | 1441 | 0.196 | 0.827 | 0.316 | 1207 | 0.991 | 0.972 | 0.981 |
| E12_best | argmax | excl232 | 0.666 | 0.908 | 61 | 0.016 | 0.021 | 0.019 | 1207 | 0.991 | 0.980 | 0.985 |
| E12_best | config | full | 0.827 | 0.941 | 1441 | 0.616 | 0.536 | 0.573 | 1207 | 0.964 | 0.976 | 0.970 |
| E12_best | config | excl232 | 0.651 | 0.908 | 61 | 0.098 | 0.009 | 0.016 | 1207 | 0.964 | 0.983 | 0.974 |
