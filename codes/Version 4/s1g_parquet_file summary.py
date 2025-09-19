import pyarrow.parquet as pq

file_path = r"E:\1. Github\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset\training_combinations_part00000.parquet"

pf = pq.ParquetFile(file_path)

# Column names
print("Columns:", pf.schema.names)

# Row count
print("Total rows:", pf.metadata.num_rows)
