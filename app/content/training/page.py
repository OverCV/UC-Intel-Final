"""
Page: Training Configuration
Self-contained page module
"""

from components import render_header, render_sidebar
from content.training.view import render

render_header()
render_sidebar()

render()
