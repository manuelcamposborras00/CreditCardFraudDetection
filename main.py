"""
Pipeline completo — Credit Card Fraud Detection
Ejecutar desde la raíz del proyecto: python main.py
"""
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent


def run_script(script_path: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {script_path.name}")
    print('='*60)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT)
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "ERROR"
    print(f"[{status}] {script_path.name} — {elapsed:.1f}s")
    return result.returncode == 0


if __name__ == '__main__':
    # Verificar dataset
    if not (PROJECT_ROOT / 'data' / 'creditcard.csv').exists():
        print("ERROR: data/creditcard.csv no encontrado.")
        print("Descárgalo de: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        sys.exit(1)

    # Crear directorios
    for d in ['logs', 'results']:
        (PROJECT_ROOT / d).mkdir(exist_ok=True)

    script_names = [
        ('01', 'eda'),
    ]

    scripts = [PROJECT_ROOT / 'src' / f'{num}_{name}.py'
               for num, name in script_names]

    results = {}
    total_start = time.time()

    for script in scripts:
        ok = run_script(script)
        results[script.name] = ok
        if not ok:
            print(f"\nERROR en {script.name}. Abortando pipeline.")
            break

    total = time.time() - total_start
    n_ok  = sum(results.values())

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETADO — {n_ok}/{len(results)} scripts OK — {total:.1f}s total")
    print('='*60)
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL'} {name}")
