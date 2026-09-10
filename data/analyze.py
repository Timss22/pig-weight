# import pandas as pd
# df = pd.read_csv("inventory.csv")

# a = set(df[df.set=="rgb9579"].folder_id)
# b = set(df[df.set=="rgb3394"].folder_id)
# print("9579 IDs:", len(a), "| 3394 IDs:", len(b), "| shared:", len(a & b))
# print("shared IDs:", sorted(a & b))

import pandas as pd
df = pd.read_csv("inventory.csv")
g = df[df.set=="rgb9579"].groupby("folder").agg(n=("path","size"), tag=("folder_id","first"))
print((g.n == g.tag).mean(), g.head(10))

import random
for f in random.sample(sorted(df[df.set=="rgb9579"].folder.unique()), 3):
    print(f, "→", (df.folder==f).sum(), "images")