"""
Page: Training Monitor
Self-contained page module
"""

from components import render_header, render_sidebar
from content.monitor.view import render

render_header()
render_sidebar()

render()
