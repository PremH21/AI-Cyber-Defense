import sys
from pymongo import MongoClient

db = MongoClient("mongodb://localhost:27017")["cyberdefense"]
col, legacy = db["incidents"], db["incidents_legacy"]
q = {"timestamp": {"$type": "string"}}
apply = len(sys.argv) > 1 and sys.argv[1] == "--apply"

n_old = col.count_documents(q)
print("total in incidents:", col.count_documents({}))
print("string timestamps (old schema):", n_old)
print("numeric timestamps (new engine):", col.count_documents({"timestamp": {"$type": ["double", "int", "long"]}}))
print("string-timestamp docs that have an attack_type:", col.count_documents({"timestamp": {"$type": "string"}, "attack_type": {"$ne": None}}))

if not apply:
    print("DRY RUN only. Re-run with --apply to move them.")
    sys.exit(0)

if legacy.count_documents({}) != 0:
    sys.exit("incidents_legacy is not empty, aborting")

docs = list(col.find(q))
legacy.insert_many(docs)
assert legacy.count_documents({}) == n_old, "copy count mismatch, nothing deleted"
col.delete_many(q)
print("moved", n_old, "docs. incidents now has", col.count_documents({}))
