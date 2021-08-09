import sys
import pandas as pd
import numpy as np
from content_based import Content

if __name__ == "__main__":
    # leitura dos argumentos de entrada
    contents_path = sys.argv[1]
    ratings_path = sys.argv[2]
    targets_path = sys.argv[3]
    content = Content()
    # content.read_content(contents_path)
    content.read_ratings(ratings_path)
    content.read_content(contents_path)
    # content.compute_user_vectors()
    # content.submission(targets_path)

