# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "grainsack"
copyright = "2025, Roberto"
author = "Roberto"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.mathjax"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

mathjax3_config = {
    "tex": {
        "macros": {
            "G": r"\mathcal{G}",
            "F": r"\mathcal{F}",
            "X": r"\mathcal{X}",
            "V": r"\mathcal{V}",
            "R": r"\mathcal{R}",
            "spo": r"\langle s, p, o \rangle",
            "query": r"\langle s, p, ?\rangle",
            "indicator": "𝟙"
        }
    }
}
