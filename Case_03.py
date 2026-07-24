import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl


def gather_data():
    dataframe = pd.read_csv("Lecture_02_Information/agro_colombia.csv")
    dataframe.describe()


if __name__ == "__main__":
    gather_data()
