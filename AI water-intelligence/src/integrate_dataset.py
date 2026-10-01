import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 1. Define paths
# ---------------------------------------------------------

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Load original INDOFLOODS datasets
# ---------------------------------------------------------

print("Loading datasets...")

flood = pd.read_csv(
    RAW / "floodevents_indofloods.csv"
)

precip = pd.read_csv(
    RAW / "precipitation_variables_indofloods.csv"
)

catchment = pd.read_csv(
    RAW / "catchment_characteristics_indofloods.csv"
)

metadata = pd.read_csv(
    RAW / "metadata_indofloods.csv"
)


# ---------------------------------------------------------
# 3. Create GaugeID from EventID
# ---------------------------------------------------------

flood["GaugeID"] = (
    flood["EventID"]
    .astype(str)
    .str.rsplit("-", n=1)
    .str[0]
)

print("\nFlood events:", flood.shape)
print("Precipitation:", precip.shape)
print("Catchment:", catchment.shape)
print("Metadata:", metadata.shape)


# ---------------------------------------------------------
# 4. Merge flood events + precipitation
# ---------------------------------------------------------

print("\nMerging flood events with precipitation...")

integrated = flood.merge(
    precip,
    on="EventID",
    how="left",
    suffixes=("", "_precip")
)

print("After precipitation merge:", integrated.shape)


# ---------------------------------------------------------
# 5. Merge catchment characteristics
# ---------------------------------------------------------

print("\nMerging catchment characteristics...")

integrated = integrated.merge(
    catchment,
    on="GaugeID",
    how="left",
    suffixes=("", "_catchment")
)

print("After catchment merge:", integrated.shape)


# ---------------------------------------------------------
# 6. Merge metadata
# ---------------------------------------------------------

print("\nMerging metadata...")

integrated = integrated.merge(
    metadata,
    on="GaugeID",
    how="left",
    suffixes=("", "_metadata")
)

print("After metadata merge:", integrated.shape)


# ---------------------------------------------------------
# 7. Check matching coverage
# ---------------------------------------------------------

print("\n========== MATCHING CHECK ==========")

print(
    "Catchment match:",
    integrated["Drainage Area"].notna().mean() * 100,
    "%"
)

print(
    "Metadata match:",
    integrated["Station"].notna().mean() * 100,
    "%"
)


# ---------------------------------------------------------
# 8. Check duplicate columns
# ---------------------------------------------------------

duplicate_columns = [
    col for col in integrated.columns
    if col.endswith("_precip")
    or col.endswith("_catchment")
    or col.endswith("_metadata")
]

print("\nDuplicate/suffixed columns:")
print(duplicate_columns)


# ---------------------------------------------------------
# 9. Save integrated dataset
# ---------------------------------------------------------

output_file = PROCESSED / "indofloods_integrated.csv"

integrated.to_csv(
    output_file,
    index=False
)

print("\n===================================")
print("INTEGRATION COMPLETE")
print("Final shape:", integrated.shape)
print("Saved to:", output_file)
print("===================================")