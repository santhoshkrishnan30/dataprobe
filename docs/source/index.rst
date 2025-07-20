Welcome to DataProbe's documentation!
=====================================

**DataProbe** is a comprehensive Python toolkit for debugging, profiling, and optimizing data pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Features
--------

* **Operation Tracking**: Track execution time, memory usage, and data shapes
* **Visual Pipeline Flow**: Generate pipeline execution visualizations  
* **Memory Profiling**: Monitor and identify memory-intensive operations
* **Data Lineage**: Track data transformations throughout the pipeline
* **Bottleneck Detection**: Identify slow operations and memory peaks
* **Performance Reports**: Generate debugging reports with optimization suggestions

Quick Example
-------------

.. code-block:: python

   from dataprobe import PipelineDebugger
   
   debugger = PipelineDebugger(name="My_Pipeline")
   
   @debugger.track_operation("Load Data")
   def load_data():
       return pd.read_csv("data.csv")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
