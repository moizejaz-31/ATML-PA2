import os

# Mitigate Windows OpenMP multiple runtime initialization conflict across torch and numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
