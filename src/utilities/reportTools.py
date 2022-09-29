import pandas as pd 
import numpy as np
import pandas_profiling 

def create_profile_report(df, save_path,  minimal=False):
    report = pandas_profiling.ProfileReport(df, minimal=minimal)
    report.to_file(output_file=save_path)
