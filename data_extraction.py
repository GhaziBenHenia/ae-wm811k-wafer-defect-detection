import os, csv, numpy as np, pandas as pd, cv2, re, shutil

df = pd.read_pickle("data/LSWMD.pkl")
print(f"Loaded dataset: {len(df)} entries")

out_dir = "extracted_images"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

def clean_label(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        x = x[0] if len(x) > 0 else "unlabeled"
    if not x or str(x).strip() == "":
        x = "unlabeled"
    return re.sub(r"[^\w\-]", "_", str(x).strip())

meta_path = os.path.join(out_dir, "metadata.csv")
with open(meta_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file", "label", "split", "lotName", "dieSize"])

    saved = 0
    for i, row in enumerate(df.itertuples(index=False)):
        wafer = np.asarray(row.waferMap).squeeze()
        label = clean_label(row.failureType)
        split = getattr(row, "trianTestLabel", "")
        lot = getattr(row, "lotName", "")
        die_size = getattr(row, "dieSize", "")

        label_dir = os.path.join(out_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        file_name = f"wafer_{i}.png"
        file_path = os.path.join(label_dir, file_name)
        cv2.imwrite(file_path, (wafer.astype(np.uint8)) * 127)

        writer.writerow([os.path.relpath(file_path, out_dir), label, split, lot, die_size])

        saved += 1
        if saved % 10000 == 0:
            print(f"Processed {saved}/{len(df)} wafers...")

print(f"Done: {saved} images and metadata.csv saved to {os.path.abspath(out_dir)}")
