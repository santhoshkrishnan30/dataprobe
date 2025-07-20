Contributing
============

We welcome contributions! Please follow these guidelines.

How to Contribute
-----------------

1. Fork the repository
2. Create your feature branch (``git checkout -b feature/AmazingFeature``)
3. Commit your changes (``git commit -m 'Add some AmazingFeature'``)
4. Push to the branch (``git push origin feature/AmazingFeature``)
5. Open a Pull Request

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/santhoshkrishnan30/dataprobe.git
   cd dataprobe
   pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   pytest
   pytest --cov=dataprobe

Code Style
----------

We use Black for code formatting:

.. code-block:: bash

   black dataprobe tests

Reporting Issues
----------------

Please report issues on `GitHub Issues <https://github.com/santhoshkrishnan30/dataprobe/issues>`_.
