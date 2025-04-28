import pandas as pd
import numpy as np

def precompute_dot_products(colors_df):
    """
    Precompute dot products between all color pairs.
    Args:
        colors_df: DataFrame containing 'color', 'r', 'g', 'b' columns.
    Returns:
        dot_products: Dictionary {(color1, color2): dot_product_value}
        max_dot_product: Maximum dot product found (for normalization)
    """
    dot_products = {}
    max_dot_product = 0

    colors = colors_df.dropna(subset=['r', 'g', 'b'])

    for i in range(len(colors)):
        for j in range(i, len(colors)):
            color1 = colors.iloc[i]
            color2 = colors.iloc[j]
            dp = np.dot(
                [color1['r'], color1['g'], color1['b']],
                [color2['r'], color2['g'], color2['b']]
            )
            dot_products[(color1['color'].lower(), color2['color'].lower())] = dp
            if dp > max_dot_product:
                max_dot_product = dp

    return dot_products, max_dot_product

def get_color_similarity(color1, color2, dot_products, max_dot_product):
    """
    Retrieve the dot product and similarity percentile between two colors.
    Args:
        color1: Name of first color (str)
        color2: Name of second color (str)
        dot_products: Dictionary of precomputed dot products
        max_dot_product: Maximum dot product value
    Returns:
        Dictionary { 'dot_product': value, 'similarity_percentile': value }
    """
    color1 = color1.lower()
    color2 = color2.lower()
    
    dp = dot_products.get((color1, color2)) or dot_products.get((color2, color1))
    if dp is None:
        return {"error": "Color pair not found."}

    similarity = (dp / max_dot_product) * 100
    return {
        "dot_product": dp,
        "similarity_percentile": round(similarity, 2)
    }
