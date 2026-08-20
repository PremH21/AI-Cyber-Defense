"""
Maps the native dataset labels (UNSW-NB15, CIC-IDS-2017) to the 9 attack
categories named in the project slide deck. This exists because public IDS
datasets use their own labeling conventions, not a fixed "ransomware / DDoS /
APT / MITM / phishing / insider / zero-day / C2 / lateral movement" taxonomy.

This mapping is descriptive, not something the models were trained on — it's
a lookup table used for reporting and for the RL response engine's
threat_type field (see response-engine/rl_response_agent.py THREAT_TYPES).
"""

import json

SLIDE_CATEGORIES = [
    "Ransomware", "DDoS", "APT", "MITM", "Phishing",
    "Insider Threats", "C2 Communication", "Lateral Movement", "Zero-Day",
]

MAPPING = {
    "UNSW-NB15": {
        "Normal":         {"slide_category": "N/A (benign)", "notes": "Not an attack; baseline traffic."},
        "Generic":        {"slide_category": "N/A (crypto probe, no clean match)", "notes": "Attacks against block ciphers regardless of structure — doesn't correspond to any single slide category. Closest conceptually to APT reconnaissance tooling."},
        "Exploits":       {"slide_category": "APT", "notes": "Exploitation of a known vulnerability — mapped to APT/intrusion since UNSW's Exploits class covers post-recon compromise attempts."},
        "Fuzzers":        {"slide_category": "Zero-Day", "notes": "Fuzzing tools probing for unknown vulnerabilities — closest conceptual match to zero-day discovery activity."},
        "DoS":            {"slide_category": "DDoS", "notes": "Denial of service — direct match (UNSW's DoS includes both single- and multi-source floods)."},
        "Reconnaissance": {"slide_category": "Lateral Movement", "notes": "Scanning/probing prior to compromise — precursor to lateral movement in the kill chain."},
        "Analysis":       {"slide_category": "Phishing", "notes": "Port scan + spam + HTML file penetration attacks — closest match is web/phishing-adjacent activity."},
        "Backdoor":       {"slide_category": "C2 Communication", "notes": "Backdoor access techniques — direct conceptual match to C2 channels."},
        "Shellcode":      {"slide_category": "Ransomware", "notes": "Small malicious code payload delivery — used as the closest stand-in for ransomware payload delivery, since UNSW predates modern ransomware-specific labeling."},
        "Worms":          {"slide_category": "Insider Threats", "notes": "Self-replicating malware — weak match; included in RL engine as malware_backdoor rather than insider. Flagged as the least clean mapping in this table."},
    },
    "CIC-IDS-2017": {
        "BENIGN":                     {"slide_category": "N/A (benign)", "notes": "Not an attack; baseline traffic."},
        "DDoS":                       {"slide_category": "DDoS", "notes": "Direct match."},
        "DoS Hulk":                   {"slide_category": "DDoS", "notes": "Direct match (single-source DoS tool)."},
        "DoS GoldenEye":              {"slide_category": "DDoS", "notes": "Direct match."},
        "DoS Slowhttptest":           {"slide_category": "DDoS", "notes": "Direct match (slow-rate DoS)."},
        "DoS slowloris":              {"slide_category": "DDoS", "notes": "Direct match (slow-rate DoS)."},
        "PortScan":                   {"slide_category": "Lateral Movement", "notes": "Reconnaissance/scanning — precursor to lateral movement."},
        "FTP-Patator":                {"slide_category": "Insider Threats", "notes": "Credential brute-force — mapped to insider/credential-abuse category (matches RL engine's credential_attack threat_type)."},
        "SSH-Patator":                {"slide_category": "Insider Threats", "notes": "Credential brute-force — same rationale as FTP-Patator."},
        "Bot":                        {"slide_category": "C2 Communication", "notes": "Botnet client traffic — direct match to C2."},
        "Infiltration":               {"slide_category": "APT", "notes": "Multi-stage internal compromise — direct match to APT-style long-dwell intrusion."},
        "Heartbleed":                 {"slide_category": "Zero-Day", "notes": "CVE-2014-0160 exploitation — direct match, this is a canonical zero-day/known-CVE case."},
        "Web Attack \ufffd Brute Force":     {"slide_category": "Phishing", "notes": "Web application attack — grouped under phishing/web-vector category."},
        "Web Attack \ufffd Sql Injection":   {"slide_category": "Phishing", "notes": "Web application attack — same rationale."},
        "Web Attack \ufffd XSS":             {"slide_category": "Phishing", "notes": "Web application attack — same rationale."},
    },
}


def main():
    print("=== Attack Taxonomy Mapping: Dataset Labels -> Slide's 9 Categories ===\n")

    md_lines = [
        "# Attack Taxonomy Mapping\n",
        "This table reconciles the native labels in UNSW-NB15 and CIC-IDS-2017 with the ",
        "9 attack categories named in the project proposal slide deck. Public IDS datasets ",
        "predate and do not follow a fixed threat-category naming scheme, so this mapping is ",
        "provided for reporting clarity — it is descriptive, not a relabeling used in training.\n",
        f"\n**Slide's named categories:** {', '.join(SLIDE_CATEGORIES)}\n",
    ]

    for dataset_name, labels in MAPPING.items():
        print(f"--- {dataset_name} ---")
        md_lines.append(f"\n## {dataset_name}\n")
        md_lines.append("| Dataset Label | Slide Category | Notes |")
        md_lines.append("|---|---|---|")
        for label, info in labels.items():
            print(f"  {label:32s} -> {info['slide_category']}")
            md_lines.append(f"| {label} | {info['slide_category']} | {info['notes']} |")
        print()

    covered = set()
    for labels in MAPPING.values():
        for info in labels.values():
            covered.add(info["slide_category"])
    covered.discard("N/A (benign)")

    uncovered = [c for c in SLIDE_CATEGORIES if c not in covered and not any(c in x for x in covered)]
    print("=== Coverage check against slide's 9 named categories ===")
    for cat in SLIDE_CATEGORIES:
        hit = any(cat == info["slide_category"] for labels in MAPPING.values() for info in labels.values())
        print(f"  {'[COVERED]' if hit else '[NOT DIRECTLY COVERED]':24s} {cat}")

    md_lines.append("\n## Coverage summary\n")
    md_lines.append("| Slide Category | Covered by dataset labels? |")
    md_lines.append("|---|---|")
    for cat in SLIDE_CATEGORIES:
        hit = any(cat == info["slide_category"] for labels in MAPPING.values() for info in labels.values())
        md_lines.append(f"| {cat} | {'Yes' if hit else 'No — no dataset label maps directly'} |")

    md_lines.append(
        "\n**Honest caveat for the report/defense:** several mappings above are the *closest "
        "conceptual match*, not a literal label rename (e.g. UNSW's `Shellcode` standing in for "
        "`Ransomware`, `Worms` for `Insider Threats`). This reflects a real limitation of academic "
        "IDS datasets: none of the widely-used ones (UNSW-NB15, CIC-IDS-2017, KDD Cup 99) use "
        "modern ransomware/APT/phishing terminology natively, since they predate or predate-adjacent "
        "to current threat naming conventions. The RL response engine (`response-engine/rl_response_agent.py`) "
        "uses its own 8-category `threat_type` taxonomy (benign, dos_ddos, recon_scan, web_attack, "
        "credential_attack, malware_backdoor, infiltration, botnet) as the operational bridge between "
        "raw classifier output and response-action selection — this is the taxonomy actually used at "
        "runtime, and it is documented and defensible on its own terms.\n"
    )

    with open("data-collection/attack_taxonomy_mapping.json", "w") as f:
        json.dump(MAPPING, f, indent=2)

    with open("data-collection/ATTACK_TAXONOMY_MAPPING.md", "w") as f:
        f.write("\n".join(md_lines))

    print("\nSaved: data-collection/attack_taxonomy_mapping.json")
    print("Saved: data-collection/ATTACK_TAXONOMY_MAPPING.md (drop this into your report)")


if __name__ == "__main__":
    main()
