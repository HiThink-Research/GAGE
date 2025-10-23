pip config set global.index-url https://pypi.org/simple
pip install sphinx
pip install sphinx_rtd_theme
pip install sphinx_autodoc_typehints

cd ..
PYTHONPATH=. sphinx-autogen docs/source/index.rst

cd docs
make html
# sphinx-autogen -o docs/source/ docs/source/index.rst