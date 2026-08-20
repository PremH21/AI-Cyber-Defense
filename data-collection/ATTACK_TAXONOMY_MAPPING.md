# Attack Taxonomy Mapping

This table reconciles the native labels in UNSW-NB15 and CIC-IDS-2017 with the 
9 attack categories named in the project proposal slide deck. Public IDS datasets 
predate and do not follow a fixed threat-category naming scheme, so this mapping is 
provided for reporting clarity — it is descriptive, not a relabeling used in training.


**Slide's named categories:** Ransomware, DDoS, APT, MITM, Phishing, Insider Threats, C2 Communication, Lateral Movement, Zero-Day


## UNSW-NB15

| Dataset Label | Slide Category | Notes |
|---|---|---|
| Normal | N/A (benign) | Not an attack; baseline traffic. |
| Generic | N/A (crypto probe, no clean match) | Attacks against block ciphers regardless of structure — doesn't correspond to any single slide category. Closest conceptually to APT reconnaissance tooling. |
| Exploits | APT | Exploitation of a known vulnerability — mapped to APT/intrusion since UNSW's Exploits class covers post-recon compromise attempts. |
| Fuzzers | Zero-Day | Fuzzing tools probing for unknown vulnerabilities — closest conceptual match to zero-day discovery activity. |
| DoS | DDoS | Denial of service — direct match (UNSW's DoS includes both single- and multi-source floods). |
| Reconnaissance | Lateral Movement | Scanning/probing prior to compromise — precursor to lateral movement in the kill chain. |
| Analysis | Phishing | Port scan + spam + HTML file penetration attacks — closest match is web/phishing-adjacent activity. |
| Backdoor | C2 Communication | Backdoor access techniques — direct conceptual match to C2 channels. |
| Shellcode | Ransomware | Small malicious code payload delivery — used as the closest stand-in for ransomware payload delivery, since UNSW predates modern ransomware-specific labeling. |
| Worms | Insider Threats | Self-replicating malware — weak match; included in RL engine as malware_backdoor rather than insider. Flagged as the least clean mapping in this table. |

## CIC-IDS-2017

| Dataset Label | Slide Category | Notes |
|---|---|---|
| BENIGN | N/A (benign) | Not an attack; baseline traffic. |
| DDoS | DDoS | Direct match. |
| DoS Hulk | DDoS | Direct match (single-source DoS tool). |
| DoS GoldenEye | DDoS | Direct match. |
| DoS Slowhttptest | DDoS | Direct match (slow-rate DoS). |
| DoS slowloris | DDoS | Direct match (slow-rate DoS). |
| PortScan | Lateral Movement | Reconnaissance/scanning — precursor to lateral movement. |
| FTP-Patator | Insider Threats | Credential brute-force — mapped to insider/credential-abuse category (matches RL engine's credential_attack threat_type). |
| SSH-Patator | Insider Threats | Credential brute-force — same rationale as FTP-Patator. |
| Bot | C2 Communication | Botnet client traffic — direct match to C2. |
| Infiltration | APT | Multi-stage internal compromise — direct match to APT-style long-dwell intrusion. |
| Heartbleed | Zero-Day | CVE-2014-0160 exploitation — direct match, this is a canonical zero-day/known-CVE case. |
| Web Attack � Brute Force | Phishing | Web application attack — grouped under phishing/web-vector category. |
| Web Attack � Sql Injection | Phishing | Web application attack — same rationale. |
| Web Attack � XSS | Phishing | Web application attack — same rationale. |

## Coverage summary

| Slide Category | Covered by dataset labels? |
|---|---|
| Ransomware | Yes |
| DDoS | Yes |
| APT | Yes |
| MITM | No — no dataset label maps directly |
| Phishing | Yes |
| Insider Threats | Yes |
| C2 Communication | Yes |
| Lateral Movement | Yes |
| Zero-Day | Yes |

**Honest caveat for the report/defense:** several mappings above are the *closest conceptual match*, not a literal label rename (e.g. UNSW's `Shellcode` standing in for `Ransomware`, `Worms` for `Insider Threats`). This reflects a real limitation of academic IDS datasets: none of the widely-used ones (UNSW-NB15, CIC-IDS-2017, KDD Cup 99) use modern ransomware/APT/phishing terminology natively, since they predate or predate-adjacent to current threat naming conventions. The RL response engine (`response-engine/rl_response_agent.py`) uses its own 8-category `threat_type` taxonomy (benign, dos_ddos, recon_scan, web_attack, credential_attack, malware_backdoor, infiltration, botnet) as the operational bridge between raw classifier output and response-action selection — this is the taxonomy actually used at runtime, and it is documented and defensible on its own terms.
