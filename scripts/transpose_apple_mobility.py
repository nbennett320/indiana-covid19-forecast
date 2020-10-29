import numpy as np
import pandas as pd
data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
df = pd.DataFrame(data)

def main():
  dft = df.T
  dft.to_csv(r'./data/applemobilitytrends-2020-10-26-indiana_transposed.csv')

main()