#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  This script is used to validate the python implementation of 
  Recommendation ITU-R P.2001 as defined in the package Py2001
  
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
          Adrien Demarez (AD)

  Revision History:
  Date            Revision
  06SEP22       Initial version (IS)
  17OCT23       Revised version using Pandas (AD)
"""

import os
import numpy as np
import pandas as pd

from Py2001 import P2001

tol = 1e-6
success = 0
total = 0

# path to the folder containing test profiles
test_profiles = "./validation_examples/"

# Collect all the filenames .csv in the folder test_profiles that contain the profile data
try:
    filenames = [f for f in os.listdir(test_profiles) if f.endswith(".csv")]
except:
    print("The system cannot find the given folder " + test_profiles)

for filename in filenames:
    if not filename.endswith("_profile.csv"):
        continue
    df1 = pd.read_csv(test_profiles + filename, skiprows=9, names=['d', 'h', 'z'])
    df2 = pd.read_csv(test_profiles + filename.replace("_profile.csv", "_results.csv"))

    print("\nProcessing file " + filename)
    GHz_prev = -1
    Tpc_last = df2.Tpc.to_numpy()[-1]
    Lb_computed = []
    for i, row in df2.iterrows():
        if row.GHz > GHz_prev:
            print(f"\tProcessing {i+1} / {len(df2)}, GHz={row.GHz}, Tpc={row.Tpc} % - {Tpc_last} % ...")
            GHz_prev = row.GHz
        Lb_computed.append( P2001.bt_loss(df1.d.to_numpy(), df1.h.to_numpy(), df1.z.to_numpy(), row.GHz, row.Tpc, row.Phire, \
                                          row.Phirn, row.Phite, row.Phitn, row.Hrg, row.Htg, row.Grx, row.Grt, row.FlagVp) )
    Lb_computed = np.array(Lb_computed)
    Lb_ref = df2.Lb
    delta = np.abs(Lb_computed - Lb_ref)

    # verify error in the results against tolerance
    if np.max(delta) < tol:
        success += 1
    else:
        (kk,) = np.where(delta > tol)
        print(delta[kk])

    total += 1

print(f"Validation results: {success} out of {total} tests passed successfully.")
if success == total:
    print(f"The deviation from the reference results is smaller than {tol}.")
