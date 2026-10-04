# A/B Testing & User Retention Analytics

An end-to-end analytics project designed to evaluate the impact of a product feature experiment on user conversion, revenue, funnel performance, and long-term user retention.

The project combines **Python, Statistical Hypothesis Testing, Cohort Retention Analysis, PostgreSQL, SQL, and Power BI** to transform raw event-level data into actionable business insights.

> **Note:** This project uses synthetically generated data for portfolio and analytical demonstration purposes. The experiment results should not be interpreted as results from a real company or production experiment.

---

## Project Overview

A simulated e-commerce platform conducted an A/B experiment to evaluate a redesigned checkout experience.

Users were randomly assigned to one of two experimental groups:

- **Control** — Existing checkout experience
- **Treatment** — Redesigned checkout experience

The objective was to determine whether the Treatment experience improved:

1. Conversion rate
2. Revenue per user
3. Purchase funnel performance
4. User retention
5. Performance across customer segments

The complete workflow was implemented from data generation and validation to statistical analysis, SQL analytics, PostgreSQL integration, and Power BI visualization.

---

## Business Problem

Product teams frequently need to determine whether a new feature or product change actually improves business performance.

Simply observing that the Treatment group has a higher conversion rate is not enough.

A robust experiment should answer:

- Did the Treatment group actually improve conversion?
- Is the improvement statistically significant?
- How large is the improvement?
- Did revenue per user increase?
- Which stage of the funnel improved?
- Did the new experience negatively affect retention?
- Does the treatment perform consistently across devices and acquisition channels?
- Is the experiment statistically valid?

This project addresses these questions through an end-to-end experimentation and retention analytics workflow.

---

# Objectives

### Primary Objectives

- Measure Control vs Treatment conversion rates
- Calculate absolute and relative conversion lift
- Perform statistical significance testing
- Calculate confidence intervals
- Analyze revenue per user
- Analyze the complete conversion funnel
- Perform weekly cohort retention analysis
- Test retention differences between Control and Treatment
- Analyze performance across customer segments
- Validate experiment randomization
- Build SQL-based analytical reporting
- Create an interactive Power BI dashboard

---

# Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Statistics | SciPy, Statsmodels |
| Visualization | Matplotlib, Seaborn, Plotly |
| Database | PostgreSQL |
| SQL | PostgreSQL SQL |
| Database Connectivity | SQLAlchemy, psycopg2 |
| BI & Dashboard | Power BI |
| Version Control | Git & GitHub |
| Environment | Python Virtual Environment |

---

# Project Architecture

```text
                    ┌─────────────────────┐
                    │ Synthetic Raw Data  │
                    │                     │
                    │ Users               │
                    │ Assignments         │
                    │ Events              │
                    │ Orders              │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Validation     │
                    │                     │
                    │ Null Checks          │
                    │ Duplicate Checks     │
                    │ Referential Integrity│
                    │ SRM Validation       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │ Analysis             │
                    │                     │
                    │ KPI Analysis         │
                    │ Funnel Analysis      │
                    │ Segment Analysis     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │ Statistical A/B Test│          │ Cohort Retention    │
    │                     │          │ Analysis             │
    │ Z-Test              │          │ Weekly Cohorts      │
    │ P-Value             │          │ Retention Matrix    │
    │ Confidence Interval │          │ Retention Testing   │
    │ Effect Size         │          │ Holm Correction     │
    └──────────┬──────────┘          └──────────┬──────────┘
               │                                │
               └────────────────┬───────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ PostgreSQL          │
                     │                     │
                     │ Analytics Schema    │
                     │ SQL KPI Analysis    │
                     │ Funnel Analysis     │
                     │ Segment Analysis    │
                     │ Cohort Analysis     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Power BI Dashboard  │
                     │                     │
                     │ KPI Cards           │
                     │ Funnel              │
                     │ Experiment Results  │
                     │ Cohort Retention    │
                     │ Segment Analysis    │
                     └─────────────────────┘
