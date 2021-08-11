import sys
import pandas as pd
import numpy as np
from content_based import Content

if __name__ == "__main__":
    # Read args
    contents_path = sys.argv[1]
    ratings_path = sys.argv[2]
    targets_path = sys.argv[3]

    # Call the content-based recommender
    content = Content()
    content.read_ratings(ratings_path)
    content.read_content(contents_path)
    content.submission(targets_path)
