from pathlib import Path
import re, pandas as pd
from PIL import Image

ROOT = Path("data/PigRGB-Weight") 
def scan(folder, label):
    rows = []
    for img in Path(folder).rglob("*.png"):
        parent = img.parent.name          # e.g. "73.36_124"
        m_w = re.match(r"([\d.]+)kg", img.name)
        m_p = re.match(r"([\d.]+)_(\d+)", parent)
        rows.append({
            "set": label,
            "path": str(img),
            "file_kg": float(m_w.group(1)) if m_w else None,
            "folder_kg": float(m_p.group(1)) if m_p else None,
            "folder_id": int(m_p.group(2)) if m_p else None,
            "folder": parent,
        })
    return pd.DataFrame(rows)

df = pd.concat([
    scan(ROOT/"RGB_9579", "rgb9579"),
    scan(ROOT/"RGB_MASK_3394/RGB_3394", "rgb3394"),
    scan(ROOT/"RGB_MASK_3394/MASK_3394", "mask3394"),
], ignore_index=True)

print(df.groupby("set").size(), "\n")

# --- THE KEY QUESTION: what is folder_id? ---
main = df[df.set == "rgb9579"]
print("unique folders:", main.folder.nunique())
print("unique folder_ids:", main.folder_id.nunique())
print("images per folder:\n", main.groupby("folder").size().describe(), "\n")
# If one folder_id maps to MANY weights -> it's likely a PIG ID (great: split by pig)
# If each folder_id has exactly one weight -> it's just a session/measurement group
print(main.groupby("folder_id").file_kg.nunique().value_counts().head(), "\n")

# --- weight distribution ---
print(main.file_kg.describe())
main.file_kg.plot(kind="hist", bins=50, title="Weight distribution (kg)").figure.savefig("weights.png")

# --- does file_kg always match folder_kg? ---
print("\nmismatch rows:", (main.file_kg != main.folder_kg).sum())

# --- image sizes (sample 30) ---
sizes = [Image.open(p).size for p in main.path.sample(min(30, len(main)), random_state=0)]
print("\nsizes seen:", set(sizes))

# --- do masks pair 1:1 with RGB? ---
r = set(Path(p).name for p in df[df.set=="rgb3394"].path)
m = set(Path(p).name for p in df[df.set=="mask3394"].path)
print("\nRGB-only:", len(r-m), " Mask-only:", len(m-r))

df.to_csv("inventory.csv", index=False)

print(df[df.set=="rgb3394"].file_kg.describe())

m = df[df.set=="rgb3394"]
print(m.folder.nunique(), m.folder_id.nunique())
print(m.groupby("folder_id").file_kg.nunique().value_counts())

a = set(df[df.set=="rgb9579"].folder_id)
b = set(df[df.set=="rgb3394"].folder_id)
print(len(a), len(b), len(a & b))