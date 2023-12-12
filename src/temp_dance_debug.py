import sys
from dance import create_dance_from_data_file, MockDanceManager

if __name__ == "__main__":
    pd = create_dance_from_data_file(sys.argv[1])
    ad = create_dance_from_data_file(sys.argv[2])
    dm = MockDanceManager(pd, ad)
    dm.compare_dances_live(0.5)

#python .\src\temp_dance_debug.py .\pattern.csv .\actual.csv
#*reopen file in windows
