# %% [markdown]
"""
Example: Exporting Benchmark Suite Metadata to LaTeX

This example demonstrates how to use the metadata property on OpenMLBenchmarkSuite
to generate LaTeX tables for academic publications.

The metadata property returns a pandas DataFrame containing both task-level and
dataset-level information, which can be easily exported to LaTeX using pandas'
Styler API.
"""

# %%
import openml

# %% [markdown]
# ## Getting Suite Metadata
#
# First, we retrieve a benchmark suite. Here we use OpenML-CC18, a curated suite
# of 72 classification tasks.

# %%
suite = openml.study.get_suite(99)  # OpenML-CC18
print(f"Suite: {suite.name}")
print(f"Number of tasks: {len(suite.tasks)}")

# %% [markdown]
# ## Accessing Metadata
#
# The `metadata` property returns a pandas DataFrame with comprehensive
# information about all tasks in the suite. This includes both task-specific
# information (like estimation procedure) and dataset characteristics (like
# number of instances and features).

# %%
metadata = suite.metadata
print(f"Metadata shape: {metadata.shape}")
print(f"\nFirst few columns: {metadata.columns.tolist()[:10]}")

# %% [markdown]
# ## Selecting Columns for Publication
#
# For a typical publication table, we might want to include:
# - Dataset name
# - Number of instances
# - Number of features
# - Number of classes
# - Number of missing values

# %%
# Select relevant columns for the table
columns = [
    "name",
    "NumberOfInstances",
    "NumberOfFeatures",
    "NumberOfClasses",
    "NumberOfMissingValues",
]

# Filter to only include columns that exist in the DataFrame
available_columns = [col for col in columns if col in metadata.columns]
table_data = metadata[available_columns]

print(f"\nSelected {len(available_columns)} columns")
print(table_data.head())

# %% [markdown]
# ## Generating LaTeX Table
#
# We use pandas' Styler API to format and export the table to LaTeX.
# The Styler provides many formatting options for professional-looking tables.

# %%
# Generate LaTeX table with formatting
latex_table = (
    table_data.style.format(
        {
            "NumberOfInstances": "{:,}",  # Add thousand separators
            "NumberOfFeatures": "{:d}",  # Integer format
            "NumberOfClasses": "{:d}",
            "NumberOfMissingValues": "{:d}",
        }
    )
    .hide(axis="index")  # Hide row indices
    .to_latex(
        caption="Dataset Characteristics for OpenML-CC18",
        label="tab:cc18_metadata",
        hrules=True,  # Add horizontal rules
        position="H",  # Float position
    )
)

print(latex_table)

# %% [markdown]
# ## Saving to File
#
# The LaTeX code can be saved directly to a file for inclusion in your document.

# %%
# Save to file
with open("suite_metadata.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("LaTeX table saved to 'suite_metadata.tex'")

# %% [markdown]
# ## Advanced Formatting
#
# For more advanced formatting, you can:
# - Apply conditional formatting
# - Add custom CSS classes
# - Format specific rows or columns
# - Include multi-level headers

# %%
# Example: Format rows with high number of missing values
def highlight_missing(row):
    """Highlight rows with many missing values."""
    if row["NumberOfMissingValues"] > 100:
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)


styled_table = (
    table_data.style.apply(highlight_missing, axis=1)
    .format(
        {
            "NumberOfInstances": "{:,}",
            "NumberOfFeatures": "{:d}",
            "NumberOfClasses": "{:d}",
            "NumberOfMissingValues": "{:d}",
        }
    )
    .hide(axis="index")
)

# Note: Styler.to_latex() doesn't support all CSS styling, but basic formatting works
latex_advanced = styled_table.to_latex(
    caption="Dataset Characteristics (Highlighted Missing Values)",
    label="tab:cc18_metadata_advanced",
    hrules=True,
)

print("Advanced LaTeX table generated")

# %% [markdown]
# ## Summary
#
# The `metadata` property makes it easy to:
# 1. Access comprehensive task and dataset information
# 2. Filter and select relevant columns
# 3. Export to LaTeX for academic publications
# 4. Apply custom formatting as needed
#
# This workflow eliminates the need for manual data aggregation and ensures
# consistency across publications using the same benchmark suite.

