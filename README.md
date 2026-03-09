# U.S. Flight Delay Analysis (2013–2023)

## Overview
This project analyzes U.S. flight delay patterns from 2013 to 2023 using airline operational data. The goal is to understand how delays are distributed across causes, airlines, airports, years, months, and days of the week.

Rather than treating delay as a single metric, this project studies flight delays from multiple dimensions:

- **cause-level decomposition** (Carrier, Weather, NAS, Security, Late Aircraft)
- **severity analysis** (delay minutes and average duration)
- **seasonal patterns**
- **airline-level operational profiles**
- **airport-level operational profiles**
- **year-over-year trend shifts**
- **weekday arrival performance**

This repository is designed as an exploratory data analysis project that converts raw airline records into interpretable operational insights.

---

## Motivation
Flight delays are one of the most visible indicators of operational efficiency in air transportation. They affect passenger experience, airline resource planning, airport congestion, aircraft rotations, and downstream scheduling.

This project aims to answer a more useful question than simply *“how late are flights?”*:

- What are the main sources of disruption?
- Which delay types are most common?
- Which delay types are most severe?
- Are delays concentrated in specific seasons or weekdays?
- Which airlines and airports show the weakest operational performance?

By structuring the analysis around these questions, the project provides a clearer picture of how delays propagate through the U.S. air transportation system.

---

## Dataset
This project uses the **US Airline Dataset** from Kaggle:

- **Dataset source:** [Flight Delay Data for U.S. Airports — Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay)

The dataset contains U.S. airline operational records and delay-related fields that enable analysis across:

- arrival delay
- carrier delay
- weather delay
- NAS delay
- security delay
- late aircraft delay
- airline-level performance
- airport-level performance
---

## Project Structure

```
ECE143-WI26-G7/
├── main_analysis.ipynb
└── src/
    ├── __pycache__/
    ├── airport_analysis.py
    ├── carrier_analysis.py
    ├── cause_analysis.py
    ├── constants.py
    ├── day_of_week.py
    ├── monthly_analysis.py
    ├── preprocessing.py
    └── yearly_change_analysis.py
├── dataset/ (auto created when running code)
├── outputs/ (auto created when running code)
```

### Directory Descriptions
- **`main_analysis.ipynb`**: Main Jupyter notebook that orchestrates the entire analysis workflow.
- **`src/`**: Source code directory with Python modules for data preprocessing, analysis functions, and constants.
- **`dataset/`**: Contains the raw and cleaned airline delay datasets used for analysis.
- **`outputs/`**: Directory for generated plots, figures, and analysis results.


---

## Project Objectives
This analysis is built to answer the following questions:

1. Which delay causes contribute the most incidents and the most total delay minutes?
2. Which delay causes are the most severe on a per-incident basis?
3. How do delay patterns vary by month and season?
4. How do delays evolve year by year?
5. Which airlines have higher carrier-delay rate or longer carrier-delay duration?
6. Which airports show the weakest delay performance among high-volume airports?
7. How does arrival delay behavior differ across days of the week?

---

## Methodology
The workflow of this project includes the following steps:

1. Load and inspect raw airline delay data
2. Clean and standardize delay-related fields
3. Aggregate delay records by:
   - cause
   - month
   - weekday
   - year
   - airline
   - airport
4. Compute descriptive metrics such as:
   - delay incidence share
   - total delay-minute share
   - average delay duration
   - delay rate
   - year-over-year percent change
5. Generate visualizations to compare patterns across multiple dimensions

This project focuses on **descriptive analytics and visual interpretation**, rather than predictive modeling.

---

## Key Findings

### 1. Late Aircraft and Carrier delays dominate total disruption
The analysis shows that **Late Aircraft** and **Carrier** are the two largest contributors to total delay minutes:

- **Late Aircraft:** **39.0%** of total delay minutes (**283.1M minutes**)
- **Carrier:** **33.9%** of total delay minutes (**246.4M minutes**)
- **NAS:** **21.7%** of total delay minutes (**157.8M minutes**)
- **Weather:** **5.2%** of total delay minutes (**38.2M minutes**)
- **Security:** **0.2%** of total delay minutes (**1.3M minutes**)

This suggests that the largest system-wide disruptions are driven less by rare external shocks and more by recurring operational propagation effects.

### 2. Weather delays are less frequent, but highly severe
Although weather contributes a relatively small share of total delay incidents, it tends to generate longer delays when it occurs. In other words, weather is not the most common source of disruption, but it is one of the most operationally expensive per event.

### 3. Delay patterns show strong seasonality
Monthly delay analysis indicates a clear **seasonal effect**, with delays generally rising during the **summer travel period**, especially in **June, July, and August**.

This likely reflects a combination of:

- heavier traffic volume
- tighter aircraft utilization
- congestion spillover
- weather-related summer disruptions

### 4. Carrier-delay behavior differs significantly by airline
The carrier delay profile plot shows meaningful differences in **delay rate**, **delay duration**, **traffic volume**, and **cancellation rate** across airlines.

Examples visible from the airline profile include:

- **Southwest** has very large flight volume with moderate carrier-delay rate and lower average duration than several peers.
- Some smaller carriers, such as **ExpressJet** and **Mesa**, show longer average carrier-delay durations.
- Airlines differ not only in *how often* delays happen, but also in *how long* they last and how strongly they correlate with cancellations.

This highlights that operational weakness is multi-dimensional and cannot be captured by a single average.

### 5. Delay trends changed sharply around 2020
The year-by-year trend plots show that delay levels fell sharply in **2020**, then rebounded afterward.

This pattern appears across multiple delay causes and reflects a major structural shift in flight activity and network behavior. After the 2020 drop, several delay categories recovered strongly in 2021–2022.

### 6. Carrier delay is volatile year over year
The carrier year-over-year percent change chart shows large swings across the timeline:

- very high jump in early years
- sharp drop in **2020**
- strong rebound in **2021**
- continued fluctuation afterward

This suggests carrier-related delays are highly sensitive to broader operational conditions and network recovery patterns.

### 7. Delay rates vary across days of the week
Weekday delay-rate analysis shows that arrival delay behavior is not evenly distributed:

- **Tuesday** has the highest share of flights delayed more than **15 minutes** (**14.1%**) and more than **30 minutes** (**10.3%**)
- **Friday** is also elevated:
  - **13.6%** delayed > 15 min
  - **9.2%** delayed > 30 min
- **Sunday** performs best:
  - **10.8%** delayed > 15 min
  - **7.1%** delayed > 30 min

This suggests weekday traffic and scheduling structure may meaningfully influence arrival reliability.

---

## Main Outputs
The project generates several visual outputs, including:

- **Delay distribution by cause**
  - share of total delay minutes
  - total delay minutes by cause

- **Carrier delay profile**
  - delay rate vs. average delay duration
  - bubble size = traffic volume
  - color = cancellation rate

- **Year-by-year delay trend by cause**
  - tracks long-term changes across Carrier, Weather, NAS, Security, and Late Aircraft

- **Year-over-year percent change**
  - highlights volatility and structural breaks, especially for carrier delays

- **Monthly delay trend**
  - shows seasonal concentration of delays

- **Weekday arrival delay rate**
  - compares share of flights delayed more than 15 or 30 minutes

- **Arrival delay breakdown by day**
  - compares early, on-time, minor, moderate, and severe arrival outcomes

- **Airport performance comparison**
  - compares busy airports by delay rate, average duration, traffic volume, and dominant cause
