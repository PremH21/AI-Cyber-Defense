import random
import tlsh
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import os
import json

OUT_DIR = "lineage/output"
NUM_FAMILIES = 4
VARIANTS_PER_FAMILY = 5
BASE_SIZE = 4096
MUTATION_RATE = 0.05  # fraction of bytes mutated per generation

def random_bytes(n):
    return bytes(random.getrandbits(8) for _ in range(n))

def mutate(data, rate):
    data = bytearray(data)
    n_mutations = int(len(data) * rate)
    for _ in range(n_mutations):
        idx = random.randrange(len(data))
        data[idx] = random.getrandbits(8)
    return bytes(data)

def generate_simulated_samples():
    samples = {}
    for fam in range(NUM_FAMILIES):
        family_name = f"family_{fam}"
        base = random_bytes(BASE_SIZE)
        current = base
        for variant in range(VARIANTS_PER_FAMILY):
            current = mutate(current, MUTATION_RATE)
            sample_name = f"{family_name}_v{variant}"
            samples[sample_name] = current
    return samples

def compute_tlsh_hashes(samples):
    hashes = {}
    for name, data in samples.items():
        h = tlsh.hash(data)
        if h and h != "TNULL":
            hashes[name] = h
        else:
            print(f"  WARNING: {name} too small/uniform for TLSH, skipping")
    return hashes

def build_distance_matrix(hashes):
    names = list(hashes.keys())
    n = len(names)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = tlsh.diff(hashes[names[i]], hashes[names[j]])
            dist[i][j] = d
            dist[j][i] = d
    return names, dist

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating simulated ransomware family variants...")
    samples = generate_simulated_samples()
    print(f"Generated {len(samples)} simulated samples across {NUM_FAMILIES} families")

    print("\nComputing TLSH hashes...")
    hashes = compute_tlsh_hashes(samples)
    print(f"Computed {len(hashes)} valid TLSH hashes")

    with open(f"{OUT_DIR}/tlsh_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)

    print("\nBuilding pairwise TLSH distance matrix...")
    names, dist = build_distance_matrix(hashes)

    print("Clustering (hierarchical, average linkage)...")
    from scipy.spatial.distance import squareform
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.title("Simulated Ransomware Family Lineage (TLSH clustering)")
    plt.xlabel("Sample")
    plt.ylabel("TLSH distance")
    plt.tight_layout()
    out_path = f"{OUT_DIR}/lineage_dendrogram.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved lineage dendrogram to {out_path}")
    print("Samples from the same simulated family should cluster together in the plot.")

if __name__ == "__main__":
    main()
