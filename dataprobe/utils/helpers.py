import pandas as pd
import polars as pl
import numpy as np
from typing import Any, Union, Tuple, Optional
import hashlib
import json
from pathlib import Path

def get_dataframe_info(df: Union[pd.DataFrame, pl.DataFrame]) -> dict:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: pandas or polars DataFrame
    
    Returns:
        Dictionary with DataFrame information
    """
    info = {
        "type": type(df).__name__,
        "shape": df.shape,
        "columns": list(df.columns),
        "memory_usage_mb": 0.0
    }
    
    if isinstance(df, pd.DataFrame):
        info["memory_usage_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
        info["dtypes"] = df.dtypes.to_dict()
        info["null_counts"] = df.isnull().sum().to_dict()
    elif isinstance(df, pl.DataFrame):
        info["dtypes"] = dict(zip(df.columns, df.dtypes))
        info["null_counts"] = {col: df[col].null_count() for col in df.columns}
    
    return info

def detect_dataframe_changes(df_before: Union[pd.DataFrame, pl.DataFrame],
                            df_after: Union[pd.DataFrame, pl.DataFrame]) -> dict:
    """
    Detect changes between two DataFrames.
    
    Args:
        df_before: DataFrame before transformation
        df_after: DataFrame after transformation
    
    Returns:
        Dictionary describing changes
    """
    changes = {
        "shape_change": {
            "before": df_before.shape,
            "after": df_after.shape
        },
        "columns_added": [],
        "columns_removed": [],
        "columns_modified": []
    }
    
    cols_before = set(df_before.columns)
    cols_after = set(df_after.columns)
    
    changes["columns_added"] = list(cols_after - cols_before)
    changes["columns_removed"] = list(cols_before - cols_after)
    
    # Check for dtype changes in common columns
    common_cols = cols_before & cols_after
    for col in common_cols:
        if isinstance(df_before, pd.DataFrame) and isinstance(df_after, pd.DataFrame):
            if df_before[col].dtype != df_after[col].dtype:
                changes["columns_modified"].append({
                    "column": col,
                    "dtype_before": str(df_before[col].dtype),
                    "dtype_after": str(df_after[col].dtype)
                })
    
    return changes

def estimate_memory_usage(obj: Any) -> float:
    """
    Estimate memory usage of an object in MB.
    
    Args:
        obj: Object to estimate memory for
    
    Returns:
        Estimated memory usage in MB
    """
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024 / 1024
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / 1024 / 1024
    elif isinstance(obj, pl.DataFrame):
        # Rough estimation for polars
        return sum(obj[col].estimated_size() for col in obj.columns) / 1024 / 1024
    else:
        # Rough estimation for other objects
        import sys
        return sys.getsizeof(obj) / 1024 / 1024

def generate_data_hash(data: Any) -> str:
    """
    Generate a hash for data to track it across transformations.
    
    Args:
        data: Data to hash
    
    Returns:
        Hash string
    """
    if isinstance(data, (pd.DataFrame, pl.DataFrame)):
        # Hash based on shape and column names
        hash_input = f"{data.shape}_{sorted(data.columns)}"
    elif isinstance(data, np.ndarray):
        hash_input = f"{data.shape}_{data.dtype}"
    else:
        hash_input = str(type(data)) + str(id(data))
    
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def format_duration(seconds: float) -> str:
    """
    Format duration in a human-readable way.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"

def save_dataframe_sample(df: Union[pd.DataFrame, pl.DataFrame], 
                         path: Path, 
                         sample_size: int = 100):
    """
    Save a sample of DataFrame for debugging.
    
    Args:
        df: DataFrame to sample
        path: Path to save the sample
        sample_size: Number of rows to sample
    """
    if isinstance(df, pd.DataFrame):
        sample = df.head(sample_size)
        sample.to_csv(path, index=False)
    elif isinstance(df, pl.DataFrame):
        sample = df.head(sample_size)
        sample.write_csv(path)

def create_performance_plot(operations: list, save_path: Path):
    """
    Create a performance visualization plot.
    
    Args:
        operations: List of operation metrics
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract data
    names = [op['name'] for op in operations]
    durations = [op['duration'] for op in operations]
    memory = [op.get('memory_delta', 0) for op in operations]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Duration plot
    colors = ['red' if d > 1.0 else 'green' for d in durations]
    ax1.barh(names, durations, color=colors)
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_title('Operation Execution Time')
    ax1.grid(True, alpha=0.3)
    
    # Memory plot
    colors = ['red' if m > 100 else 'blue' for m in memory]
    ax2.barh(names, memory, color=colors)
    ax2.set_xlabel('Memory Delta (MB)')
    ax2.set_title('Memory Usage Change')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def validate_dataframe(df: Union[pd.DataFrame, pl.DataFrame]) -> dict:
    """
    Validate a DataFrame and return issues found.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary of validation results
    """
    issues = {
        "critical": [],
        "warnings": [],
        "info": []
    }
    
    # Check for empty DataFrame
    if len(df) == 0:
        issues["critical"].append("DataFrame is empty")
        return issues
    
    # Check for duplicated columns
    if len(df.columns) != len(set(df.columns)):
        issues["critical"].append("Duplicate column names found")
    
    # Check for high null percentage
    if isinstance(df, pd.DataFrame):
        null_percentages = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentages[null_percentages > 50].to_dict()
        if high_null_cols:
            issues["warnings"].append(f"High null percentage in columns: {high_null_cols}")
    
    # Check for potential memory issues
    if isinstance(df, pd.DataFrame):
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 1000:
            issues["warnings"].append(f"Large DataFrame: {memory_mb:.2f}MB")
    
    # Check for object dtypes that could be categorized
    if isinstance(df, pd.DataFrame):
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                issues["info"].append(f"Column '{col}' could be converted to category type")
    
    return issues
