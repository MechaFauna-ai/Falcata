Development Guide
=================

Algorithms
----------

Refer to `Features <./Features.rst>`__ for understanding of important algorithms used in Falcata.

Classes and Code Structure
--------------------------

Important Classes
~~~~~~~~~~~~~~~~~

+-------------------------+----------------------------------------------------------------------------------------+
| Class                   | Description                                                                            |
+=========================+========================================================================================+
| ``Application``         | The entrance of application, including training and prediction logic                   |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Bin``                 | Data structure used for storing feature discrete values (converted from float values)  |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Boosting``            | Boosting interface (GBDT, DART, etc.)                                                  |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Config``              | Stores parameters and configurations                                                   |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Dataset``             | Stores information of dataset                                                          |
+-------------------------+----------------------------------------------------------------------------------------+
| ``DatasetLoader``       | Used to construct dataset                                                              |
+-------------------------+----------------------------------------------------------------------------------------+
| ``FeatureGroup``        | Stores the data of feature, could be multiple features                                 |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Metric``              | Evaluation metrics                                                                     |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Network``             | Network interfaces and communication algorithms                                        |
+-------------------------+----------------------------------------------------------------------------------------+
| ``ObjectiveFunction``   | Objective functions used to train                                                      |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Tree``                | Stores information of tree model                                                       |
+-------------------------+----------------------------------------------------------------------------------------+
| ``TreeLearner``         | Used to learn trees                                                                    |
+-------------------------+----------------------------------------------------------------------------------------+

Code Structure
~~~~~~~~~~~~~~

+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Path                | Description                                                                                                                        |
+=====================+====================================================================================================================================+
| ./include           | Header files                                                                                                                       |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./include/utils     | Some common functions                                                                                                              |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/application   | Implementations of training and prediction logic                                                                                   |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/boosting      | Implementations of Boosting                                                                                                        |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/io            | Implementations of IO related classes, including ``Bin``, ``Config``, ``Dataset``, ``DatasetLoader``, ``Feature`` and ``Tree``     |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/metric        | Implementations of metrics                                                                                                         |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/network       | Implementations of network functions                                                                                               |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/objective     | Implementations of objective functions                                                                                             |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/treelearner   | Implementations of tree learners                                                                                                   |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+

Documents API
-------------

Refer to `docs README <./README.rst>`__.

C API
-----

Refer to `C API <./C-API.rst>`__ or the comments in `c\_api.h <https://github.com/MechaFauna-ai/Falcata/blob/master/include/Falcata/c_api.h>`__ file, from which the documentation is generated.

Tests
-----

C++ unit tests are located in the ``./tests/cpp_tests`` folder and written with the help of Google Test framework.
To run tests locally first refer to the `Installation Guide <./Installation-Guide.rst#build-c-unit-tests>`__ for how to build tests and then simply run compiled executable file.
It is highly recommended to build tests with `sanitizers <./Installation-Guide.rst#sanitizers>`__.

Python tests are located in ``./tests/python_package_test`` and ``./tests/c_api_test``. The CUDA tests among them skip unless ``TASK=cuda`` is set.
With `pytest-xdist <https://pytest-xdist.readthedocs.io/>`__ installed, the CUDA suite can run in several processes that share one GPU:

.. code:: sh

  TASK=cuda python -m pytest ./tests/python_package_test ./tests/c_api_test -n 4 --dist worksteal

On an RTX 5090 with 24 CPU cores (WSL2), the full CUDA suite took 18 minutes serially and 9 minutes with ``-n 4``.
More workers did not help (10 minutes with ``-n 3``, 9 with ``-n 5``): the processes take turns on the GPU, and most tests spend their time in small CUDA calls, which get slower when several processes share the device.
Host memory use peaked at about 24, 27 and 28 GB with ``-n 3``, ``-n 4`` and ``-n 5``, because some tests build multi-GB datasets, in subprocesses too.

High Level Language Package
---------------------------

See the implementations at `Python-package <https://github.com/MechaFauna-ai/Falcata/tree/master/python-package>`__ and `R-package <https://github.com/MechaFauna-ai/Falcata/tree/master/R-package>`__.

Questions
---------


Also feel free to open `issues <https://github.com/MechaFauna-ai/Falcata/issues>`__ if you met problems.
