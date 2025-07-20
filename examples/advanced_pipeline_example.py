"""
Advanced example showing complex pipeline debugging scenarios
"""

import pandas as pd
import polars as pl
import numpy as np
from dataprobe import PipelineDebugger
import asyncio

# Initialize debugger with custom settings
debugger = PipelineDebugger(
    name="Advanced_ETL_Pipeline",
    track_memory=True,
    track_lineage=True,
    memory_threshold_mb=25,
    save_path=Path("./debug_advanced")
)

# Example with error handling
@debugger.track_operation("Risky Operation")
def risky_operation(df: pd.DataFrame) -> pd.DataFrame:
    """Operation that might fail."""
    if len(df) > 500:
        # Simulate an error condition
        raise ValueError("DataFrame too large for this operation")
    return df

# Example with nested operations
@debugger.track_operation("Parent Operation")
def parent_operation(df: pd.DataFrame) -> pd.DataFrame:
    """Parent operation with nested tracked operations."""
    
    @debugger.track_operation("Nested Operation 1")
    def nested_op1(data):
        return data * 2
    
    @debugger.track_operation("Nested Operation 2")
    def nested_op2(data):
        return data + 10
    
    df['value'] = nested_op1(df['value'])
    df['value'] = nested_op2(df['value'])
    
    return df

# Example with polars DataFrame
@debugger.track_operation("Polars Processing")
def process_with_polars(data: dict) -> pl.DataFrame:
    """Example using Polars for high-performance processing."""
    # Create Polars DataFrame
    df = pl.DataFrame(data)
    
    # Perform operations
    df = df.with_columns([
        (pl.col("value") * 2).alias("doubled"),
        pl.col("category").str.to_uppercase().alias("category_upper")
    ])
    
    return df

# Example with validation
@debugger.track_operation("Validate and Process")
def validate_and_process(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data before processing."""
    from dataprobe.utils import validate_dataframe
    
    # Validate DataFrame
    issues = validate_dataframe(df)
    
    if issues['critical']:
        raise ValueError(f"Critical validation issues: {issues['critical']}")
    
    if issues['warnings']:
        print(f"Warnings: {issues['warnings']}")
    
    # Process data
    df['validated'] = True
    
    return df

# Main execution for advanced example
if __name__ == "__main__":
    print("Starting Advanced PipelineDebugger Example\n")
    
    # Create sample data
    sample_data = {
        'id': range(100),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(sample_data)
    
    # Test nested operations
    df = parent_operation(df)
    
    # Test with Polars
    polars_df = process_with_polars(sample_data)
    
    # Test validation
    df = validate_and_process(df)
    
    # Test error handling
    try:
        large_df = pd.DataFrame({'value': range(1000)})
        risky_operation(large_df)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Show final summary
    debugger.print_summary()
    
    print("\nAdvanced example completed!")
