"""
Page: Results & Evaluation
Self-contained page module
"""

from components import render_header, render_sidebar
from content.results.view import render

render_header()
render_sidebar()

render()
