# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#import unittest.mock as mock
sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, os.path.abspath('../src/direpack/'))


# -- Project information -----------------------------------------------------

project = 'direpack'
copyright = '2021, Sven Serneels and Emmanuel Jordy Menvouta'
author = 'Sven Serneels and Emmanuel Jordy Menvouta'

# The full version, including alpha/beta/rc tags
release = '1.0.10'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage','sphinx.ext.autosummary', 'sphinx.ext.napoleon','sphinx.ext.imgmath', "sphinx.ext.viewcode", 'sphinx_math_dollar']

# Add autosummary
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


 
# MOCK_MODULES = ['numpy', 'pandas', 'matplotlib','scikit-learn']
# for mod_name in MOCK_MODULES:
#      sys.modules[mod_name] = mock.Mock()


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


imgmath_latex_preamble = r'''
\usepackage{lineno}
\usepackage{amsmath}
\usepackage{graphicx,psfrag,epsf}
\usepackage{enumerate} 
\usepackage{amsmath,amsfonts,amssymb,graphicx,multirow}
\usepackage{mdsymbol}
\usepackage{booktabs}
\usepackage{amsthm}
\usepackage{bbm}
\usepackage{algorithm}
\newcommand{\argmax}{\mathop{\mbox{argmax}}}
\usepackage[noend]{algpseudocode}
\usepackage{rotating}
\modulolinenumbers[5]
\def\independenT#1#2{\mathrel{\rlap{$#1#2$}\mkern2mu{#1#2}}}
\def\spacingset#1{\renewcommand{\baselinestretch}%
{#1}\small\normalsize} \spacingset{1}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
'''

#imgmath_image_format = 'svg'
