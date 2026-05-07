End-to-end CLV Segmentation Engine for FMCG D2C brands · RFM scoring + BG/NBD CLV prediction + K-Means clustering · Python · Power BI · BRD · FRD · RTM# CLV Segmentation Engine — FMCG D2C Brand


### End-to-End Requirements & Analytics Deliverable

> **Domain:** FMCG / D2C Personal Care | **Role modelled:** Business Analyst / Product Owner  
> **Stack:** Python 3.10+ · Pandas · Lifetimes (BG/NBD) · Scikit-learn · Power BI  
> **Author:** Amit Pattanaik | [LinkedIn](#) | [Email](mailto:pattnaikamit25@gmail.com)

---

## Business problem

A mid-size FMCG D2C brand (personal care) with 280,000 registered customers sends the **same promotional email to every customer**, regardless of purchase behaviour. Marketing spend is ₹2.4 Cr/quarter at a blended ROAS of **1.8×** — well below the 3.0× category benchmark. High-value loyalists receive the same 15% discount as one-time buyers, eroding margin. Lapsed high-CLV customers receive no win-back intervention.

**This project defines, documents, and implements a Customer Lifetime Value Segmentation Engine** that classifies customers into 5 behavioural tiers and enables personalised, segment-specific campaign execution.

---

## Repository structure

```
clv-segmentation-engine/
│
├── README.md                          ← You are here
│
├── docs/
│   ├── BRD.md                         ← Business Requirements Document
│   ├── FRD.md                         ← Functional Requirements Document
│   ├── NFRD.md                        ← Non-Functional Requirements Document
│   └── RTM.md                         ← Requirements Traceability Matrix
│
├── src/
│   └── clv_rfm_model.py               ← Full Python model (RFM + BG/NBD + K-Means)
│
├── data/
│   └── sample_transactions.csv        ← Synthetic 18-month transaction dataset
│
└── outputs/
    ├── CLV_Segmentation_Dashboard.png ← 8-panel analytical chart
    ├── CLV_Segmentation_customers.csv ← Customer-level scored output (Power BI input)
    ├── CLV_Segmentation_segment_profile.csv
    └── CLV_Segmentation_campaign_playbook.csv
```

---

## Key deliverables

| Artifact | Type | Purpose |
|---|---|---|
| [BRD](docs/BRD.md) | Business doc | 18 business requirements across 4 objectives |
| [FRD](docs/FRD.md) | Technical doc | 32 functional requirements across 7 modules |
| [NFRD](docs/NFRD.md) | Technical doc | 24 non-functional requirements (performance, security, compliance) |
| [RTM](docs/RTM.md) | Traceability doc | Full BR → FR → test case → UAT status mapping |
| [Python model](src/clv_rfm_model.py) | Code | RFM scoring + BG/NBD CLV prediction + K-Means clustering |
| Power BI dashboard | Report | 6-panel KPI dashboard (see `/outputs`) |

---

## Model results summary

| Segment | Customers | % of base | Revenue share | Avg CLV (12m) | P(alive) |
|---|---|---|---|---|---|
| Champions | ~420 | 5% | 36% | ₹1,233 | 90% |
| Loyalists | ~1,830 | 22% | 36% | ₹345 | 73% |
| At-Risk | ~1,526 | 18% | 14% | ₹78 | 29% |
| Hibernating | ~1,844 | 22% | 6% | ~₹0 | ~0% |
| Lost | ~2,781 | 33% | 8% | ~₹0 | ~0% |

> **Key insight:** Top 5% of customers drive 36% of revenue but were receiving the same blanket discount as one-time buyers — direct margin erosion of ~₹22L/quarter.

---

## How to run

```bash
# 1. Clone repository
git clone https://github.com/your-username/clv-segmentation-engine.git
cd clv-segmentation-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model
python src/clv_rfm_model.py

# Outputs written to /outputs directory
```

**Requirements:**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lifetimes==0.11.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

---

## Frameworks applied

`RFM Scoring` · `BG/NBD Model` · `Gamma-Gamma Model` · `K-Means Clustering`  
`Silhouette Validation` · `ABC Customer Analysis` · `Pareto 80/20 Rule`  
`Jobs-to-be-Done` · `MoSCoW Prioritisation` · `AARRR Funnel` · `North Star Metric`

---

## Documentation map

```
Business need (BRD)
    └── Functional solution (FRD)
            └── Quality constraints (NFRD)
                    └── Test coverage (RTM)
                                └── UAT sign-off
```

---

*This is a portfolio project using synthetic data. Business figures are illustrative and designed to reflect realistic FMCG D2C scenarios.*
