
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict

# === 1) USTAW ŚCIEŻKI ===
PREDS_PATH   = Path(r"runs3\wheat_student\eval2\preds_raw.jsonl")
CLASSES_PATH = Path(r"runs3\wheat_student\eval2\classes.json")  # polecane; jak nie ma, spróbuje zgadnąć

# === 2) TABELKA PODPISÓW (oryginalna_nazwa -> podpis_na_wykresie) ===
DISPLAY_NAME: Dict[str, str] = {
    "Alt": "Altern.",
    "Big": "Sucha(Lb)",
    "Bot": "Szara",
    "Mac": "Sucha(Lm)",
    "Mil": "Rzekomy",
    "Myc": "Pierśc.",
    "Pse": "Biała",
    "Xan": "Bakt.",
    "healthy": "Zdrowe",
    "Healthy": "Zdrowe",
    "LeafBlight": "Septorioza",
    "Rust": "Rdza",
    "PowderyMildew": "Mączniak",
    "FusariumFootRot": "Fuzarioza",
    "WheatBlast": "Zaraza",
    "BlackPoint": "Czarna plam."
}




def load_classes():
    if CLASSES_PATH.exists():
        data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "classes" in data and isinstance(data["classes"], list):
                return data["classes"]
            if "idx_to_class" in data:
                itc = data["idx_to_class"]
                # bywa dict z kluczami "0","1",...
                if isinstance(itc, dict):
                    return [itc[str(i)] for i in range(len(itc))]
                if isinstance(itc, list):
                    return itc
    # fallback: z jsonl (mniej pewne)
    classes = {}
    with PREDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("y_true") is not None and r.get("y_true_name"):
                classes[int(r["y_true"])] = r["y_true_name"]
            if r.get("y_pred") is not None and r.get("y_pred_name"):
                classes[int(r["y_pred"])] = r["y_pred_name"]
    n = max(classes.keys()) + 1 if classes else 0
    return [classes.get(i, str(i)) for i in range(n)]


def plot_cm(cm, labels, title, out_path, annotate_fmt="{:.2f}"):
    n = len(labels)
    fig_w = max(7.5, 0.85 * n)
    fig_h = max(7.0, 0.85 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Bez colorbara (ten pasek z boku na screenie)
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Predykcja", fontsize=12)
    ax.set_ylabel("Prawdziwa klasa", fontsize=12)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    # adnotacje jak na screenie
    thresh = np.nanmax(cm) * 0.5 if np.isfinite(cm).any() else 0.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = annotate_fmt.format(val) if np.isfinite(val) else "nan"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val >= thresh else "black", fontsize=10)

    ax.set_ylim(n - 0.5, -0.5)  # fix dla matplotlib (żeby nie ucinało wiersza)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not PREDS_PATH.exists():
        raise SystemExit(f"Brak pliku: {PREDS_PATH.resolve()}")

    classes = load_classes()
    if not classes:
        raise SystemExit("Nie udało się ustalić listy klas (dodaj classes.json albo sprawdź preds_raw.jsonl).")

    # podpisy do wykresu
    labels = [DISPLAY_NAME.get(c, c) for c in classes]

    y_true, y_pred = [], []
    with PREDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("y_true") is None:
                continue
            y_true.append(int(r["y_true"]))
            y_pred.append(int(r["y_pred"]))

    n = len(classes)
    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    # row-normalized
    row_sum = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts.astype(np.float32), row_sum, where=row_sum != 0)

    out_dir = PREDS_PATH.parent / "relabel_cm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # zapis wykresu w stylu jak na screenie
    plot_cm(cm_norm, labels, "Macierz pomyłek", out_dir / "cm_norm.png", annotate_fmt="{:.2f}")

    # (opcjonalnie) też counts, przydatne do raportu
    plot_cm(cm_counts.astype(np.float32), labels, "Macierz pomyłek (liczności)", out_dir / "cm_counts.png", annotate_fmt="{:.0f}")

    # report.txt z nowymi podpisami
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    (out_dir / "report.txt").write_text(rep, encoding="utf-8")

    # żebyś miał “źródło prawdy”
    (out_dir / "labels_used.json").write_text(json.dumps({
        "classes_original": classes,
        "labels_display": labels,
        "preds_path": str(PREDS_PATH),
        "classes_path": str(CLASSES_PATH) if CLASSES_PATH.exists() else None
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK:", out_dir.resolve())
    print(" - cm_norm.png")
    print(" - cm_counts.png")
    print(" - report.txt")
    print(" - labels_used.json")


if __name__ == "__main__":
    main()