# __init__.py
# This file marks the 'services' directory as a Python package.
# You can optionally import your main service functions here for cleaner imports.

from .sequence_analyzer import analyze_sequences
from .color_similarity import precompute_dot_products, get_color_similarity
