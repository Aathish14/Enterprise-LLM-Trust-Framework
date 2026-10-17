import os, re
d = r".venv\Lib\site-packages\mlflow\store\db_migrations\versions"
revisions = {}
downrefs = set()
for fn in os.listdir(d):
    if not fn.endswith(".py") or fn == "__init__.py":
        continue
    txt = open(os.path.join(d, fn), encoding="utf-8").read()
    m = re.search(r"^revision\s*=\s*['\"]([^'\"]+)['\"]", txt, re.M)
    dm = re.search(r"^down_revision\s*=\s*(['\"])(.*?)\1", txt, re.M)
    if m:
        rev = m.group(1)
        revisions[rev] = fn
        if dm and dm.group(2):
            downrefs.add(dm.group(2))
heads = [r for r in revisions if r not in downrefs]
print("All revisions count:", len(revisions))
print("HEAD revision(s):", heads)
for h in heads:
    print("  ", h, "->", revisions[h])
