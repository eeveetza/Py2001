# Python Implementation of Recommendation ITU-R P.2001

This code repository contains a python software implementation of  [Recommendation ITU-R P.2001-4](https://www.itu.int/rec/R-REC-P.2001/en) with a general purpose wide-range terrestrial propagation model in the frequency range 30 MHz to 50 GHz.    

This is a development code and not necessarily in line with the original reference [MATLAB/Octave Implementation of Recommendation ITU-R P.2001](https://github/eeveetza/p2001) approved by ITU-R Working Party 3M and published on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx).

The package can be downloaded and installed using:
~~~
python -m pip install "git+https://github.com/eeveetza/Py2001@dev#egg=Py2001"   
~~~

and imported as follows
~~~
from Py2001 import P2001
~~~




| File/Folder               | Description                                                         |
|----------------------------|---------------------------------------------------------------------|
|`/src/Py2001/P2001.py`                | python implementation of Recommendation ITU-R P.2001-4         |
|`/tests/validateP2001.py`          | python script used to validate the implementation of Recommendation ITU-R P.2001-4 in `P2001.bt_loss()`             |
|`/tests/validation_profiles/`    | Folder containing a set of terrain profiles and inputs for validation of software implementations  of this Recommendation |


## Function Call


~~~
Lb = P2001.bt_loss(d, h, z, GHz, Tpc, Phire, Phirn, Phite, Phitn, Hrg, Htg, Grx, Gtx, FlagVP)

~~~ 

## Required input arguments of function `P2001.bt_loss`

| Variable          | Type   | Unit | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `d`               | array double | km   | 0 < `max(d)` ≤ ~1000 | Terrain profile distances (in the ascending order from the transmitter)|
| `h`          | array double | m (asl)   |   | Terrain profile heights |
| `z`          | array int    |       |  1 - Sea, 3 - Coastal Land, 4 - Inland |  Zone code |
| `GHz`               | scalar double | GHz   | 0.3 ≤ `GHz` ≤ 50 | Frequency   |
| `Tpc`               | scalar double | %   | 0 < `Tpc` < 100 | Percentage of time (average year) for which the predicted basic transmission loss is not exceeded |
| `Phire`               | scalar double | deg   | -180 ≤ `Phire` ≤ 180 | Receiver longitude, positive to east   |
| `Phirn`               | scalar double | deg   | -90 ≤ `Phirn` ≤ 90 | Receiver latitude, positive to north     |
| `Phite`               | scalar double | deg   | -180 ≤ `Phite` ≤ 180 | Transmitter longitude, positive to east   |
| `Phitn`               | scalar double | deg   | -90 ≤ `Phitn` ≤ 90   | Transmitter latitude, positive to north     |
| `Hrg`                 | scalar double    | m      |   0 < `hrg`  < ~8000          |  Receiving antenna height above ground |
| `Htg`                 | scalar double    | m      |   0 < `htg`  < ~8000          |  Transmitting antenna height above ground |
| `Grg`                 | scalar double    | dBi      |                             |  Receiving antenna gain in the direction of the ray to the transmitting antenna |
| `Gtg`                 | scalar double    | dBi      |            |  Transmitting antenna gain in the direction of the ray to the receiving antenna |
| `FlagVP`                 | scalar int    |        |   1, 0         |  Signal polarisation: 1 - vertical, 0 - horizontal |

## Outputs ##

| Variable   | Type   | Unit | Description |
|------------|--------|-------|-------------|
| `Lb`    | double | dB    | Basic transmission loss not exceeded Tpc % time |




## Software Versions
The code was tested and runs on:
* python3.9

## References

* [Recommendation ITU-R P.2001](https://www.itu.int/rec/R-REC-P.2001/en)

* [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx)

* [MATLAB/Octave Implementation of Recommendation ITU-R P.2001](https://github/eeveetza/p2001)
