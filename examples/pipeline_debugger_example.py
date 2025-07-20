"""
Example usage of DataAlchemy PipelineDebugger
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from dataprobe import PipelineDebugger

# Initialize debugger
debugger = PipelineDebugger(
    name="ETL_Pipeline_Example",
    track_memory=True,
    track_lineage=True,
    memory_threshold_mb=50
)

# Example 1: Basic usage with decorators
@debugger.track_operation("Load Data")
def load_data(file_path: str) -> pd.DataFrame:
    """Simulate loading data from a file."""
    # Create sample data
    data = {
        'customer_id': range(1000),
        'transaction_amount': np.random.uniform(10, 1000, 1000),
        'transaction_date': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'payment_method': np.random.choice(['Credit Card', 'Cash', 'Digital Wallet'], 1000)
    }
    df = pd.DataFrame(data)
    
    # Simulate some processing time
    time.sleep(0.5)
    
    return df

@debugger.track_operation("Clean Data")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the data."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (simulate some)
    df.loc[np.random.choice(df.index, 50), 'product_category'] = np.nan
    df['product_category'].fillna('Unknown', inplace=True)
    
    # Add derived columns
    df['transaction_hour'] = df['transaction_date'].dt.hour
    df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6])
    
    # Simulate processing time
    time.sleep(0.3)
    
    return df

@debugger.track_operation("Transform Data")
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply business transformations."""
    # Calculate transaction metrics
    df['amount_category'] = pd.cut(
        df['transaction_amount'],
        bins=[0, 50, 200, 500, 1000],
        labels=['Small', 'Medium', 'Large', 'Very Large']
    )
    
    # Add customer lifetime value (simulated)
    customer_ltv = df.groupby('customer_id')['transaction_amount'].sum().reset_index()
    customer_ltv.columns = ['customer_id', 'lifetime_value']
    df = df.merge(customer_ltv, on='customer_id', how='left')
    
    # Simulate a slow operation
    time.sleep(1.5)  # This will trigger bottleneck detection
    
    return df

@debugger.track_operation("Aggregate Data")
def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated views."""
    # Daily aggregations
    daily_stats = df.groupby([
        pd.Grouper(key='transaction_date', freq='D'),
        'product_category'
    ]).agg({
        'transaction_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
    
    # Simulate memory-intensive operation
    # Create a large temporary DataFrame to trigger memory warning
    temp_df = pd.DataFrame(np.random.randn(1000000, 10))
    del temp_df  # Clean up
    
    return daily_stats

@debugger.track_operation("Save Results")
def save_results(df: pd.DataFrame, output_path: str) -> None:
    """Save processed data."""
    # In real scenario, this would save to file/database
    print(f"Would save {len(df)} rows to {output_path}")
    time.sleep(0.2)

# Example 2: Using context manager for profiling
def example_with_profiling():
    """Example using memory profiling decorator."""
    
    @debugger.profile_memory
    def memory_intensive_operation():
        # Create large DataFrame
        large_df = pd.DataFrame(np.random.randn(1000000, 50))
        
        # Perform operations
        result = large_df.groupby(large_df.index % 1000).mean()
        
        return result
    
    result = memory_intensive_operation()
    return result

# Example 3: DataFrame analysis
def analyze_dataframe_example(df: pd.DataFrame):
    """Example of DataFrame analysis feature."""
    debugger.analyze_dataframe(df, name="Transaction Data")

# Main execution
if __name__ == "__main__":
    print("Starting DataAlchemy PipelineDebugger Example\n")
    
    # Run the pipeline
    try:
        # Load data
        df = load_data("dummy_path.csv")
        
        # Analyze initial data
        analyze_dataframe_example(df)
        
        # Process through pipeline
        df = clean_data(df)
        df = transform_data(df)
        aggregated = aggregate_data(df)
        
        # Save results
        save_results(aggregated, "output/results.csv")
        
        # Run profiling example
        profiled_result = example_with_profiling()
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
    
    # Generate visualizations and reports
    print("\nGenerating pipeline visualization...")
    debugger.visualize_pipeline()
    
    print("\nPipeline Summary:")
    debugger.print_summary()
    
    print("\nGenerating detailed report...")
    report = debugger.generate_report()
    
    print("\nOptimization Suggestions:")
    suggestions = debugger.suggest_optimizations()
    for suggestion in suggestions:
        print(f"- [{suggestion['type'].upper()}] {suggestion['operation']}: {suggestion['suggestion']}")
    
    print("\nExporting data lineage...")
    lineage_json = debugger.export_lineage(format="json")
    print(f"Lineage data (first 500 chars):\n{lineage_json[:500]}...")
    
    print("\nExample completed! Check the debug output directory for visualizations and logs.")
