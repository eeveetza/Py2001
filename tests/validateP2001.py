# -*- coding: utf-8 -*-
"""
  This script is used to validate the python implementation of 
  Recommendation ITU-R P.2001 as defined in the package Py2001
  
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
  Revision History:
  Date            Revision
  06SEP22       Initial version (IS)  
  
"""
import csv
import os
from traceback import print_last
import numpy as np
import sys

from  Py2001 import P2001


tol = 1e-6
success = 0
total = 0

# path to the folder containing test profiles
test_profiles = './validation_examples/'


# begin code
# Collect all the filenames .csv in the folder test_profiles that contain the profile data
try:
    filenames = [f for f in os.listdir(test_profiles) if f.endswith('.csv')]
except:
    print ("The system cannot find the given folder " + test_profiles)
    
for filename1 in filenames:
    
    # skip reading the results file
    
    if(filename1.find('results')!= -1):
        continue
        
    
    print ('***********************************************\n')
    print ('Processing file '  + filename1 + '\n')
    print ('***********************************************\n')
    
    failed = False
    
    # read the path profiles
    
    rows = []
    with open(test_profiles + filename1, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
            
    x = np.mat(rows)
    
    d = np.double(x[8:,0])
    d = np.squeeze(np.asarray(d))
    h = np.double(x[8:,1])
    h = np.squeeze(np.asarray(h))
    z = np.double(x[8:,2])
    z = np.squeeze(np.asarray(z))
    
  
    # read the input arguments and reference values
    filename2 = filename1[0:-12] + '_results.csv'
    
        
    rows = []
    with open(test_profiles + filename2, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    
           
    x = np.mat(rows)
    (nrows, ncols) = x.shape
    GHz = np.zeros(nrows)
    Tpc = np.zeros(nrows)
    Phire = np.zeros(nrows)
    Phirn = np.zeros(nrows)
    Phite = np.zeros(nrows)
    Phitn = np.zeros(nrows)
    Hrg = np.zeros(nrows)
    Htg = np.zeros(nrows)
    Grx = np.zeros(nrows)
    Gtx = np.zeros(nrows)
    FlagVP = np.zeros(nrows, dtype='int')
    Profile = np.empty(nrows, dtype='object')
    Lb_ref = np.zeros(nrows)
    Lba_ref = np.zeros(nrows)
    Lbes1_ref = np.zeros(nrows)
    Lbes2_ref = np.zeros(nrows)
    Lbfs_ref = np.zeros(nrows)
    Lbm1_ref = np.zeros(nrows)
    Lbm2_ref = np.zeros(nrows)
    Lbm3_ref = np.zeros(nrows)
    Lbm4_ref = np.zeros(nrows)
    Lbs_ref = np.zeros(nrows)
    Ldba_ref = np.zeros(nrows)
    Ldbka_ref = np.zeros(nrows)
    Ldbks_ref = np.zeros(nrows)
    Ldbs_ref = np.zeros(nrows)
    Ldsph_ref = np.zeros(nrows)
    Ld_ref = np.zeros(nrows)
    
    Lb_ref = np.zeros(nrows)
    delta = np.zeros(nrows)
    
    A1_ref = np.zeros(nrows)
    Fwvr_ref = np.zeros(nrows) 
    Awrsur_ref = np.zeros(nrows) 
    Awsur_ref = np.zeros(nrows) 
    Agsur_ref = np.zeros(nrows)
    
     
    for i in range(0, nrows):
        GHz[i] =    np.double(x[i,1])
        Tpc[i] =    np.double(x[i,10])
        Phire[i] =  np.double(x[i,6])
        Phirn[i] =  np.double(x[i,7])
        Phite[i] =  np.double(x[i,8])
        Phitn[i] =  np.double(x[i,9])
        Hrg[i] =    np.double(x[i,4])
        Htg[i] =    np.double(x[i,5])
        Grx[i] =    np.double(x[i,2])
        Gtx[i] =    np.double(x[i,3])
        FlagVP[i] = int(x[i,0])
        Profile[i] = x[i,11]
        
        Lb_ref[i] =    np.double(x[i,78])
        Lba_ref[i] =    np.double(x[i,79])
        Lbes1_ref[i] =    np.double(x[i,80])
        Lbes2_ref[i] =    np.double(x[i,81])
        Lbfs_ref[i] =    np.double(x[i,82])
        Lbm1_ref[i] =    np.double(x[i,83])
        Lbm2_ref[i] =    np.double(x[i,84])
        Lbm3_ref[i] =    np.double(x[i,85])
        Lbm4_ref[i] =    np.double(x[i,86])
        Lbs_ref[i] =    np.double(x[i,87])
        Ld_ref[i] =    np.double(x[i,88])
        Ldba_ref[i] =    np.double(x[i,89])
        Ldbka_ref[i] =    np.double(x[i,90])
        Ldbks_ref[i] =    np.double(x[i,91])
        Ldbs_ref[i] =    np.double(x[i,92])
        Ldsph_ref[i] =    np.double(x[i,93])
        
        
        A1_ref[i] = np.double(x[i,17])
        Fwvr_ref[i] = np.double(x[i,53])
        Awrsur_ref[i] = np.double(x[i,33])
        Awsur_ref[i] = np.double(x[i,36]) 
        Agsur_ref[i] = np.double(x[i,25])
            
        Lb = np.zeros(nrows)

             
    for i in range(0,nrows):
    #for i in range(nrows-1,nrows):
        if i == 0:
            print('Processing ' + str(i+1) + '/' + str(nrows) + ', GHz = ' + str(GHz[i]) + ' GHz,  Tpc = ' + str(Tpc[i]) + '% - ' + str(Tpc[-1]) + '% ...')
        
        if i > 0:
            if(GHz[i]>GHz[i-1]):
                print('Processing ' + str(i) + '/' + str(nrows) + ', GHz = ' + str(GHz[i]) + ' GHz,  Tpc = ' + str(Tpc[i]) + '% - ' + str(Tpc[-1]) + '% ...')
       
        Lb[i] = P2001.bt_loss(d, h, z, GHz[i], Tpc[i], Phire[i], Phirn[i], Phite[i], Phitn[i], \
            Hrg[i], Htg[i], Grx[i], Gtx[i], FlagVP[i])
         
        delta[i] = abs(Lb[i] - Lb_ref[i])
        
    #print(delta)
  
  
    # verify error in the results out1 against tolarance

    maxi = max(np.abs(delta))
    if maxi > tol:    
        (kk,) = np.where(np.abs(delta) > tol)
        if ~P2001.isempty(kk):
            print('%20s \n' %('Lb'))
            for kki in kk:
                print('%d %20g\n' %(kki, np.abs(Lb[kki]-Lb_ref[kki])))
            
            failed = True
            
    
    if (not failed):
       success = success + 1
       
    total = total + 1
    
print('Validation results: %d out of %d tests passed successfully.\n' %(success, total))
if (success == total):
    print('The deviation from the reference results is smaller than %g.\n' %(tol))       