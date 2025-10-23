.. llm-eval documentation master file, created by
   sphinx-quickstart on Fri Jan 17 23:03:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

llm-eval documentation
======================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

.. 目录整理在这里
.. toctree::
   :maxdepth: 3
   :caption: Contents:

   main

.. toctree::
   :maxdepth: 3

   inference

.. toctree::
   :maxdepth: 3

   tools

.. toctree::
   :maxdepth: 3

   utils

------------------

同花顺多模态大模型评估框架

Installation
------------

.. code-block:: bash

   pip install your-project

Quick Start
--------------------

简单的使用示例...

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. 模块配置文件自动生成
Modules
=======
.. autosummary::
   :toctree: main
   
   run

.. autosummary::
   :toctree: inference

   inference.data_server
   inference.predict_multi_gpu
   inference.predict_async
   inference.preprocess

.. autosummary::
   :toctree: tools

   tools.HttpRequest
   tools.PaasSubmitter
   tools.ExternalApi

.. autosummary::
   :toctree: utils

   utils.eval_asr
   utils.eval_mmmu
   utils.eval_TextVQA_VAL
   utils.eval_MathVista_MINI
   utils.eval_next_word_probability
   utils.eval_ocr_bench