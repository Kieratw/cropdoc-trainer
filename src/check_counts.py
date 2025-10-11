from pathlib import Path
from torchvision import datasets, transforms

def dump(split):
    ds = datasets.ImageFolder(str(Path("data_merged")/split), transforms.ToTensor())
    counts = {cls:0 for cls in ds.classes}
    for _, y in ds.samples:
        cls = ds.classes[y]
        counts[cls]+=1
    print(f"\n[{split}] {len(ds.samples)} obrazów")
    for k,v in counts.items():
        print(f"{k:30s} {v}")

if __name__=="__main__":
    for s in ["train","val","test"]:
        dump(s)
