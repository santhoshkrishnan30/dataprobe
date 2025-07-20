Installation
============

Requirements
------------

* Python 3.8 or higher
* pip

Install from PyPI
-----------------

.. code-block:: bash

   pip install dataprobe

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/santhoshkrishnan30/dataprobe.git
   cd dataprobe
   pip install -e .

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/santhoshkrishnan30/dataprobe.git
   cd dataprobe
   pip install -e ".[dev]"

Verify Installation
-------------------

.. code-block:: python

   from dataprobe import PipelineDebugger
   print("DataProbe installed successfully!")
