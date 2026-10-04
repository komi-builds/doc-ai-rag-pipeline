# Demo: NIST SP 800-61 Rev. 3

Source: [NIST SP 800-61r3](nist.sp.800-61r3.pdf), *Incident Response Recommendations and Considerations for Cybersecurity Risk Management* (public domain, US Government).

Indexed into **96 chunks** (~26,921 tokens). Model: `claude-sonnet-5-5`, effort `low`, context budget 2000 tokens.

| # | Mode | Question | Context sent | vs. whole document | Cost |
|---|------|----------|-------------:|-------------------:|-----:|
| 1 | ask | What are the CSF 2.0 Functions and how do they map to incident response? | ~1,722 tok | 6% | n/a |
| 2 | ask | Who should be involved in incident response besides the security team? | ~1,635 tok | 6% | n/a |
| 3 | ask | What should an organization do after an incident to improve? | ~1,644 tok | 6% | n/a |
| 4 | ask | How should incidents be prioritized? | ~1,427 tok | 5% | n/a |
| 5 | agent | Compare the role of the Govern function with the Respond function in incident handling. | ~1,777 tok | 7% | n/a |

*Answers not generated yet: run `python scripts/demo.py --answer` with an API key.*

## 1. What are the CSF 2.0 Functions and how do they map to incident response?

*Mode: `ask`*

<details><summary>Retrieved sources</summary>

- **Executive Summary** (p.9), rerank score 7.30
- **1. Introduction > 1.1. Purpose and Scope** (p.11), rerank score 5.51
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.1. Incident Response Life Cycle Model** (p.13-14), rerank score 5.47
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.1. Incident Response Life Cycle Model** (p.12-13), rerank score 4.24
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management** (p.18), rerank score 3.89

</details>

## 2. Who should be involved in incident response besides the security team?

*Mode: `ask`*

<details><summary>Retrieved sources</summary>

- **2. Incident Response as Part of Cybersecurity Risk Management > 2.2. Incident Response Roles and Responsibilities** (p.15-16), rerank score 3.11
- **Executive Summary** (p.9), rerank score 0.74
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management > 3.2. Incident Response** (p.38), rerank score 0.20

</details>

## 3. What should an organization do after an incident to improve?

*Mode: `ask`*

<details><summary>Retrieved sources</summary>

- **Executive Summary** (p.9), rerank score 0.08
- **1. Introduction** (p.10-11), rerank score -0.31
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.1. Incident Response Life Cycle Model** (p.14), rerank score -0.98
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.2. Incident Response Roles and Responsibilities** (p.14-15), rerank score -1.67
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.3. Incident Response Policies, Processes, and Procedures** (p.17), rerank score -2.71

</details>

## 4. How should incidents be prioritized?

*Mode: `ask`*

<details><summary>Retrieved sources</summary>

- **2. Incident Response as Part of Cybersecurity Risk Management > 2.3. Incident Response Policies, Processes, and Procedures** (p.17), rerank score 0.29
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management > 3.2. Incident Response** (p.35-36), rerank score -0.16
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management > 3.2. Incident Response** (p.38), rerank score -1.97
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management > 3.2. Incident Response** (p.37-38), rerank score -3.03
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management > 3.1. Preparation and Lessons Learned** (p.27-28), rerank score -3.68

</details>

## 5. Compare the role of the Govern function with the Respond function in incident handling.

*Mode: `agent`*

<details><summary>Retrieved sources</summary>

- **Executive Summary** (p.9), rerank score 1.01
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.1. Incident Response Life Cycle Model** (p.13), rerank score 0.12
- **3. CSF 2.0 Community Profile for Cyber Incident Risk Management > 3.2. Incident Response** (p.34), rerank score -1.42
- **2. Incident Response as Part of Cybersecurity Risk Management > 2.2. Incident Response Roles and Responsibilities** (p.14-15), rerank score -2.07

</details>
