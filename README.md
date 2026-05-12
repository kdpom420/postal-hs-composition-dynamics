# Country-level HS Composition Dynamics for Postal-friendly Export Goods


K.D. Pom   
**Date:** 2026-05-12  
**DOI:** [10.5281/zenodo.20131776](http://doi.org/10.5281/zenodo.20131776)  


## Overview

This repository contains exploratory analysis code and visualization outputs for country-level HS composition dynamics using Korea export data filtered by postal-friendly HS candidate groups.

The analysis focuses on yearly changes in HS share distributions across partner countries.

---

## Main Features

- Country-year HS share distribution construction
- Jensen-Shannon similarity calculation between consecutive years
- HS share delta decomposition
- Country-level structural transition analysis
- Consecutive yearly visualization
- Drift vs JS similarity comparison

---

## Data Source

- UN Comtrade
- HS4-level product categories

---

## Analysis Concept

For each country and year:

```text
Country × Year → HS Share Distribution
```

## Included Analysis
- HS share normalization
- Consecutive-year JS similarity
- HS increase/decrease detection
- Country-level composition change visualization

## Visualization Examples
- HS share delta barplots
- Consecutive yearly transition plots
- Country-level structural comparison
- Sankey-based HS flow visualization
- Drift vs JS scatterplots


## Notes

This repository is intended for exploratory analysis and visualization purposes.

The outputs describe structural changes observed in trade composition data and should not be interpreted as economic forecasts.