import sys, os, time, shlex, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======= ABSOLUTNIE BEZ --img-size =======
PYTHON_EXE   = sys.executable
MAX_PARALLEL = 4
OMP_THREADS  = 2

VAL_RATIO    = 0.15
TEST_RATIO   = 0.10

# Pełne ścieżki do skryptów builderów
B_RAPESEED = r"C:\Users\wojci\source\repos\repos\cropdoc-trainer\src\build_rapeseed.py"
B_WHEAT    = r"C:\Users\wojci\source\repos\repos\cropdoc-trainer\src\build_wheat.py"
B_POTATO   = r"C:\Users\wojci\source\repos\repos\cropdoc-trainer\src\build_potato.py"
B_TOMATO   = r"C:\Users\wojci\source\repos\repos\cropdoc-trainer\src\build_tomato.py"

# Roboczy katalog (żeby relative importy nie płakały)
CWD = r"C:\Users\wojci\source\repos\repos\cropdoc-trainer\src"

# Ścieżki datasetów
RAPESEED_SRC = r"D:\inzynierka\Datasety\Rzepak\dataverse_files\B.napus_Rom\Mono"
RAPESEED_OUT = r"D:\inzynierka\Datasety_do_treningu\rapeseed"

WHEAT_SRC_A  = r"D:\inzynierka\Datasety\Zboże\Original_Dataset"
WHEAT_SRC_B  = r"D:\inzynierka\Datasety\Zboże\wfd_dataset\wfd_dataset"
WHEAT_CSV    = r"D:\inzynierka\Datasety\Zboże\wfd_dataset_info\data.csv"
WHEAT_OUT    = r"D:\inzynierka\Datasety_do_treningu\wheat"
WHEAT_SCHEMA = "A7"   # lub "B5"

POTATO_SRC   = r"D:\inzynierka\Datasety\Ziemniak\ziemniak"
POTATO_OUT   = r"D:\inzynierka\Datasety_do_treningu\potato"

TOMATO_SRC   = r"D:\inzynierka\Datasety\Pomidor\cross1212"
TOMATO_OUT   = r"D:\inzynierka\Datasety_do_treningu\tomato"
TOMATO_CV    = 1      # ustaw None, jeśli wskazujesz konkretny Cross-validationX

LOGS_DIR     = r"C:\Users\wojci\source\repos\repos\cropdoc-trainer\src\build_logs"
# =========================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_one(name: str, cmd: list, log_path: Path, env: dict):
    t0 = time.monotonic()
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8", newline="") as log:
        log.write(f"== TASK: {name}\n== CWD : {CWD}\n== CMD : {' '.join(shlex.quote(x) for x in cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=CWD)
        rc = proc.wait()
        dt = time.monotonic() - t0
        log.write(f"\n== EXIT CODE: {rc}\n== DURATION : {dt:.1f} s\n")
    return name, rc, dt, log_path

def main():
    for f in [B_RAPESEED, B_WHEAT, B_POTATO, B_TOMATO]:
        if not Path(f).is_file():
            raise FileNotFoundError(f"Brak pliku: {f}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(OMP_THREADS)
    env["MKL_NUM_THREADS"] = str(OMP_THREADS)
    env["OPENBLAS_NUM_THREADS"] = str(OMP_THREADS)

    logs_dir = Path(LOGS_DIR)
    ensure_dir(logs_dir)

    tasks = []
    # Rzepak (bez --img-size; każdy builder ma własny default)
    tasks.append(("rapeseed",
        [PYTHON_EXE, "-u", B_RAPESEED,
         "--src", RAPESEED_SRC, "--out", RAPESEED_OUT,
         "--val-ratio", str(VAL_RATIO), "--test-ratio", str(TEST_RATIO)],
        logs_dir / "rapeseed.log"))

    # Pszenica
    tasks.append(("wheat",
        [PYTHON_EXE, "-u", B_WHEAT,
         "--src_a", WHEAT_SRC_A, "--src_b", WHEAT_SRC_B, "--csv", WHEAT_CSV,
         "--out", WHEAT_OUT,
         "--schema", WHEAT_SCHEMA,
         "--val-ratio", str(VAL_RATIO), "--test-ratio", str(TEST_RATIO)],
        logs_dir / "wheat.log"))

    # Ziemniak
    tasks.append(("potato",
        [PYTHON_EXE, "-u", B_POTATO,
         "--src", POTATO_SRC, "--out", POTATO_OUT,
         "--val-ratio", str(VAL_RATIO), "--test-ratio", str(TEST_RATIO)],
        logs_dir / "potato.log"))

    # Pomidor (ma tylko val; test bierze z CV)
    t_cmd = [PYTHON_EXE, "-u", B_TOMATO,
             "--src", TOMATO_SRC, "--out", TOMATO_OUT,
             "--val-ratio", str(VAL_RATIO)]
    if TOMATO_CV is not None:
        t_cmd += ["--cv", str(TOMATO_CV)]
    tasks.append(("tomato", t_cmd, logs_dir / "tomato.log"))

    print("== Absolute/relative paths OK, odpalam. ==")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(MAX_PARALLEL))) as ex:
        futs = [ex.submit(run_one, n, c, lp, env) for (n, c, lp) in tasks]
        for f in as_completed(futs):
            n, rc, dt, lp = f.result()
            results.append((n, rc, dt, lp))
            print(f"[DONE] {n:8s} {'OK' if rc==0 else f'FAIL({rc})':10s} {dt:.1f}s  log: {lp}")

    print("\n=== SUMMARY ===")
    ok = True
    for (n, rc, _, lp) in results:
        print(f"{n:8s} -> {'OK' if rc==0 else 'FAIL'} ({lp})")
        ok = ok and (rc==0)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()