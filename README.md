# Economic valuation of urban nature's service towards mental health (depression)

Estimates preventable depression cases (and their avoided societal cost) attributable
to urban nature exposure, following the InVEST-style Urban Mental Health design doc
(Yingjie Li). Compares 2019 land-use NDVI-based nature exposure against a
counterfactual no-vegetation scenario, applies a dose-response relative-risk model,
and aggregates results by urban region and country.

## Pipeline
1. Resample/warp LULC (GLC_FCS30D 30m) onto the population (WorldPop 100m) grid.
2. Convert LULC to NDVI via a land-cover attribute table.
3. Build a no-vegetation counterfactual NDVI scenario.
4. Compute change in nature exposure (baseline − no-vegetation).
5. Compute preventable depression cases per pixel (population, prevalence, effect size).
6. Rasterize urban regions onto the population grid (zone_id raster + lookup).
7. Sum preventable cases per region blockwise on that grid (each pixel counted once;
   sub-pixel regions get the free pixels they touch).
8. Apply country-specific cost per case (USD 2019) for avoided cost estimates.

## Data (not tracked in this repo — see `.gitignore`)
- WorldPop 100m population raster.
- GLC_FCS30D land-cover raster + `glc_fcs30d_attribute_table_processed.csv`.
- Urban boundaries: GUPPD levels 30/23/22 + GHS-SMOD classes 21 and 13, built using preprocessing/guppd.ipynb.
- `effect_size.xlsx` (depression odd ratio, Liu et al. 2023).
- Country-level treatment cost table.