Drift-kit:
A small novel easy to use Dataset Metadata + Drift Tracker library.
(In the space a lot of other adjacent tooling exist but nothing with high ease of use and specializing in data drift: For example Dagster, ML Flow, etc.)

Initial features:
- Ingest a Parquet file where we store dataset metadata, record file path, row count
- Compute feature stats like mean, std, min, max, missing ratio
- Drift detection where we can compare data set 'a' vs data set 'b', output drift scores, highlight largest changes, store drift metrics.

Practicality:
“What changed in the dataset?”
“What’s the difference between version 12 and version 13?”
“Why did the model accuracy drop?”
“Did someone change the schema?”
“Is drift high?”
“How can I quickly inspect Parquet files?”

Initial focus:
Convenient library. No external to postgres tooling. Simply the best design for people to build on top of and integrate with pre-existing platforms. Just connect to the libary and get immense drift feature sets and integrate it into your infrastructure.

The core of drift kit does all computation within python data structures; Parquet is the first-class ingest format. Eventually we will have systems for transitioning all internal data into new formats to be ingested into other integrations.

Everything is 0 dependency for now.
