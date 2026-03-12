# One-Click MWT ETL

This is a standalone replacement for the four notebook ETL flow:

1. `Step1_Choreography_script.ipynb`
2. `Step2_Tap_Screen_Response_Data_Analysis.ipynb`
3. `Step3_Tap_Baseline_Data_Analyses.ipynb`
4. `Step4_data_processing_for_db.ipynb`

It keeps the original notebooks untouched and moves the workflow into one CLI entrypoint with shared helper functions.

## What it does

- Optionally runs `Chore.jar` over all raw `.zip` files.
- Builds tap-response data from `.trv` files.
- Builds baseline and post-stimulus outputs from `.dat` files.
- Merges tap and PSA outputs.
- Computes plate-level summaries, MSD tables, and t-stat tables.
- Writes all outputs to one folder.
- Optionally uploads final tables to PostgreSQL using `database.ini`.
- Database upload supports:
  - `append` with `ON CONFLICT DO NOTHING`
  - `replace` using native `pandas.DataFrame.to_sql(..., if_exists="replace")`

## Run

From the repo root:

```bash
python3 -m one_click_etl
```

You will be prompted once at startup for:

- raw data root folder
- output folder
- screen from the built-in list, or a custom screen name
- whether to run `Chore.jar`
- whether to adjust more advanced settings
- whether to upload to PostgreSQL

`Chore.jar` is always resolved from the project folder at `Desktop/one_click_etl/Chore.jar`.

Advanced settings are optional. If users do not opt in, the pipeline uses these defaults:

- `number_of_taps = 30`
- `isi_seconds = 10`
- `first_tap_seconds = 600`
- `java_heap =` automatically detected from the local machine's total RAM when `Chore.jar` is used

If users choose `replace` for the database upload, they must:

- select exactly which tables to overwrite
- confirm `REPLACE`
- confirm the number of selected tables
- retype the exact selected table list

After that, the pipeline runs unattended.

## Outputs

The command writes these files into the selected output folder:

- `<SCREEN>_tap_output.csv`
- `<SCREEN>_baseline_output.csv`
- `<SCREEN>_post_stimulus.csv`
- `tap_response_data.csv`
- `tap_baseline_data.csv`
- `tstat_gene_data.csv`
- `tstat_allele_data.csv`
- `gene_MSD.csv`
- `allele_MSD.csv`
- `psa_summarised_data.csv`

## Notes

- The pipeline assumes the same experiment folder layout the notebooks used.
- For `Neuron_Genes_Screen`, `N2_N2` and `N2_XJ1` are treated as control datasets.
- Database uploads support both `append` with `ON CONFLICT DO NOTHING` and `replace` via native pandas `to_sql()`.
- `Replace` mode only runs on the specific tables the user selects and requires triple confirmation before proceeding.
