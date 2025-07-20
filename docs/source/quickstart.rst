Quick Start Guide
=================

Basic Usage
-----------

Initialize the Debugger
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataprobe import PipelineDebugger
   import pandas as pd

   # Create a debugger instance
   debugger = PipelineDebugger(
       name="My_ETL_Pipeline",
       track_memory=True,
       track_lineage=True
   )

Track Operations
~~~~~~~~~~~~~~~~

Use decorators to track your pipeline operations:

.. code-block:: python

   @debugger.track_operation("Load Data")
   def load_data(file_path):
       return pd.read_csv(file_path)

   @debugger.track_operation("Transform Data")
   def transform_data(df):
       df['new_column'] = df['value'] * 2
       return df

Generate Reports
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run your pipeline
   df = load_data("data.csv")
   df = transform_data(df)

   # View results
   debugger.print_summary()
   debugger.visualize_pipeline()
   report = debugger.generate_report()

Memory Profiling
----------------

Profile memory-intensive operations:

.. code-block:: python

   @debugger.profile_memory
   def memory_intensive_operation():
       large_df = pd.DataFrame(np.random.randn(1000000, 50))
       return large_df.groupby(large_df.index % 1000).mean()
