"""
Valida el detector de R-peaks sobre todos los registros de INCART (fs=257 Hz, resample a 360 Hz).
Imprime el resumen global (sensibilidad y PPV) al final.
"""
import os
from glob import glob
import wfdb
from src.data.rpeaks import validate_all_records

# Paso 1: Descargar INCART si no existe
data_dir = "data/raw/incartdb"
os.makedirs(data_dir, exist_ok=True)
hea_files = glob(os.path.join(data_dir, "I*.hea"))
if len(hea_files) < 75:
    print("Descargando INCART...")
    wfdb.dl_database('incartdb', dl_dir=data_dir)
    hea_files = glob(os.path.join(data_dir, "I*.hea"))

# Paso 2: Listar registros INCART
def get_incart_records(data_dir):
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in glob(os.path.join(data_dir, "I*.hea"))])

records = get_incart_records(data_dir)
print(f"Registros INCART encontrados: {len(records)}")

# Paso 3: Validar el detector (usa resample a 360 Hz internamente)
summary = validate_all_records(
    data_dir=data_dir,
    records_list=records
)

print("\nResumen global INCART:")
print(summary)
