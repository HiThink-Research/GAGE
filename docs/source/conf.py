# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'llm-eval'
copyright = '2025, Damien'
author = 'Damien'
release = '2.0.0'
language = 'zh_CN'

# sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('..'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints'
]

autodoc_default_options = {
    'members': True,  # 包含所有成员
    'member-order': 'bysource',  # 按源码顺序排列
    'special-members': '__init__',  # 包含 __init__
    'undoc-members': True,  # 包含未文档化的成员
    'exclude-members': '__weakref__'  # 排除特定成员
}

html_theme = 'sphinx_rtd_theme'
napoleon_google_docstring = True