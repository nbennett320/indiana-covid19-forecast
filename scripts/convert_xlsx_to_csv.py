import pandas as pd
from sys import argv

def main():
  if len(argv) == 3:
    xlsx_path = argv[1]
    output_path = argv[2]
    xlsx_data = pd.read_excel(xlsx_path)
    xlsx_data.to_csv(output_path, index=False)
  else:
    print("improper arguments passed, format is:\npython3 convert_xlsx_to_csv.py ./input/path.xlsx ./output/path.csv")

main()