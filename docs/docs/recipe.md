# Plumbing

This is a general overview of how OpenAVMKit is organized and how data fundamentally flows through it.

(This article is still a work in progress -- more to come soon)

In **OpenAVMKit**, all the functions you need to run the notebooks are organized in the **`pipeline` module**, located at `openavmkit/pipeline.py`.

```
openavmkit/
├──other_directories/
├──other_modules.py
├──pipeline.py # central module containing the public functions
```

In Python, a _module_ is simply a `.py` file that groups together functions, classes, and constants to make the library work as intended.

By looking into `openavmkit/pipeline.py`, you’ll find the list of **public functions** that notebooks rely on, along with their parameters (the inputs they need to run) and references to where supporting functionality is defined.

- Initializing & Syncing the notebooks:
	- ‎`init_notebook()`:
		- Initialize the notebook environment for a specific locality.
	- `load_settings()`:
		- Load and return the settings dictionary for the locality.
	- `cloud_sync()`:
		- Synchronize local files to cloud storage.
	- `from_checkpoint()`:
		- Read cached data from a checkpoint file or generate it via a function.
	- `write_checkpoint()`:
		- Write data to a checkpoint file.
	- `delete_checkpoints()`:
		- Delete all checkpoints that match the given prefix.
	- `read_pickle()`:
		- Read and return data from a pickle file.
	- `write_notebook_output_sup()`:
		- Write notebook output to disk.
- Data ETL (Extract, Transform & Load)
	- `load_dataframes()`:
		- Load dataframes based on the provided settings and return them in a dictionary.
		- As seen in: **Assemble Notebook**.
	- `process_data()`:
		- Process raw dataframes according to settings and return a SalesUniversePair.
		- As seen in: **Assemble Notebook**.
	- `process_sales()`:
		- Process sales data within a SalesUniversePair.
		- As seen in: **Assemble Notebook**, **Clean Notebook**.
	- `load_and_process_data()`:
		- Load and process data according to provided settings.
	- `load_cleaned_data_for_modeling()`:
		- Read and return the cleaned data from notebook 2.
		- As seen in: **Model Notebook**.
	- `enrich_sup_streets()`:
		- Enrich a GeoDataFrame with street network data.
		- As seen in: **Assemble Notebook**.
	- `enrich_sup_spatial_lag()`:
		- Enrich the sales and universe DataFrames with spatial lag features.
		- As seen in: **Model Notebook**.
	- `tag_model_groups_sup()`:
		- Tag model groups for a SalesUniversePair.
		- As seen in: **Assemble Notebook**.
	- `fill_unknown_values_sup()`:
		- Fill unknown values with default values as specified in settings.
		- As seen in: **Clean Notebook**.
	- `read_sales_univ()`:
		- Creates a SalesUniversePair from an existing checkpoint.
		- As seen in: **Assessment Quality Notebook**.
- Checking that the data is correct:
	- `examine_df()`:
		- Print examination details of the dataframe.
		- As seen in: **Assemble Notebook**.
	- `examine_df_in_ridiculous_detail()`:
		- Print details of the dataframe, but in RIDICULOUS DETAIL.
		- As seen in: **Assemble Notebook**.
	- `examine_sup()`:
		- Print examination details of the sales and universe data from a SalesUniversePair.
		- As seen in: **Assemble Notebook**, **Clean Notebook**, **Model Notebook**.
	- `examine_sup_in_ridiculous_detail()`:
		- Print details of the sales and universe data from a SalesUniversePair, but in RIDICULOUS DETAIL.
		- As seen in: **Assemble Notebook**.
- Clustering
	- `mark_ss_ids_per_model_group_sup()`:
		- Cluster parcels for a sales scrutiny study by assigning sales scrutiny IDs.
		- As seen in: **Clean Notebook**.
	- `mark_horizontal_equity_clusters_per_model_group_sup()`:
		- Cluster parcels for a horizontal equity study by assigning horizontal equity cluster IDs.
		- As seen in: **Clean Notebook**.
	- `run_sales_scrutiny()`:
		- Run sales scrutiny analysis for each model group within a SalesUniversePair.
		- As seen in: **Clean Notebook**.
	- `run_sales_scrutiny_per_model_group_sup()`:
		- Run sales scrutiny analysis for each model group within a SalesUniversePair.
- Modeling
	- `write_canonical_splits()`:
		- Separates data from the sales DataFrame into training and test sets, and stores the keys to disk.
		- As seen in: **Model Notebook**.
	- `try_variables()`:
		- Run tests on variables to figure out which might be the most predictive.
		- It can also print a PDF report to disk by setting the parameter "do_report" to True. _Generating PDF report requires previous installation of the wkhtmltopdf library_
		- As seen in: **Model Notebook**.
	- `try_models()`:
		- Tries out predictive models on the given SalesUniversePair. Optimized for speed and iteration, doesn't finalize results or write anything to disk.
		- As seen in: **Model Notebook**.
	- `run_models()`:
		- Runs predictive models on the given SalesUniversePair, taking detailed instructions from the provided settings dictionary.
	- `finalize_models()`:
		- Tries out predictive models on the given SalesUniversePair, finalizes results and writes to disk.
		- As seen in: **Model Notebook**.
- Evaluating the Assessment Quality
	- `run_and_write_ratio_study_breakdowns()`:
		- Run ratio study breakdowns and write the results to disk.
		- _Generating PDF report requires previous installation of the wkhtmltopdf library_
		- As seen in: **Model Notebook**.
	- `run_ratio_study()`:
		- Runs a Ratio Study for the designated time period.
		- As seen in: **Assessment Quality Notebook**.
	- `run_horizontal_equity_study()`:
		- Runs a Horizontal Equity Study for each cluster.
		- As seen in: **Assessment Quality Notebook**.
	- `run_vertical_equity_study()`:
		- Runs a Vertical Equity Study for the designated time period.
		- As seen in: **Assessment Quality Notebook**.
	- `plot_prediction_vs_sales()`:
		- Visualizes in a scatterplot the prediction from the assessment versus the actual sales.
		- As seen in: **Assessment Quality Notebook**.
