Examples
========

Complete Pipeline Example
-------------------------

.. code-block:: python

   from dataprobe import PipelineDebugger
   import pandas as pd
   import numpy as np

   # Initialize debugger
   debugger = PipelineDebugger(
       name="Sales_Analysis_Pipeline",
       track_memory=True,
       track_lineage=True,
       memory_threshold_mb=100
   )

   @debugger.track_operation("Load Sales Data")
   def load_sales_data():
       # Simulate loading data
       data = {
           'date': pd.date_range('2024-01-01', periods=1000),
           'sales': np.random.uniform(100, 1000, 1000),
           'region': np.random.choice(['North', 'South', 'East', 'West'], 1000)
       }
       return pd.DataFrame(data)

   @debugger.track_operation("Calculate Metrics")
   def calculate_metrics(df):
       df['moving_avg'] = df['sales'].rolling(window=7).mean()
       df['cumulative_sales'] = df['sales'].cumsum()
       return df

   @debugger.track_operation("Generate Report")
   def generate_report(df):
       summary = df.groupby('region')['sales'].agg(['sum', 'mean', 'count'])
       return summary

   # Run pipeline
   df = load_sales_data()
   df = calculate_metrics(df)
   report = generate_report(df)

   # View debugging information
   debugger.print_summary()
   debugger.visualize_pipeline()

   # Export lineage
   lineage = debugger.export_lineage(format="json")
   print(lineage)

Error Handling Example
----------------------

.. code-block:: python

   @debugger.track_operation("Risky Operation")
   def risky_operation(df):
       if len(df) > 500:
           raise ValueError("DataFrame too large")
       return df

   try:
       result = risky_operation(large_df)
   except ValueError as e:
       print(f"Operation failed: {e}")
       # Error is tracked in debugger
