# Install instructions to get Py2001 running on Ubuntu/Win10 machine with VS Code

1.  `git clone` this repo (a fork of [https://github.com/eeveetza/Py2001]):

```
    $ git clone https://github.com/joshuamhtsang/Py2001.git
```

2.  Open VS Code in the repo directory:  

```
    $ code .
```


3.  In Command Palette (usually crtl + shift + P), install a new venv.

4.  Open a terminal in VS Code and ensure it's running the new venv environment.  If not, `source` it.

5.  In the terminal, run (as explained in the `README.md`):

```
    $ python -m pip install "git+https://github.com/eeveetza/Py2001/#egg=Py2001"
```

You should get the message at the end: `Successfully installed Py2001-4.0`.

6.  Download the necessary maps from the ITU website:
    [https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2001-4-202109-S!!ZIP-E.zip]

    Extract it and copy into:  `./src/Py2001/maps/`
    
7.   Run the map script, first `cd` into `./src/Py2001/`

```
    $ python initiate_digital_maps.py
```
You should get the message: `P2001.npz file created successfully.`

8.  Copy `P2001.npz` into the `.venv` directory:  Due to the fact that we're running in a `venv`, we need to do an extra little step otherwise you get an error like:

```
FileNotFoundError: [Errno 2] No such file or directory: '/home/joshua/Documents/Py2001/.venv/lib/python3.12/site-packages/Py2001/P2001.npz'
>>> exit()
```
Copy the `*.npz` file over using:

```
    $ cp ./src/Py2001/P2001.npz ./.venv/lib/python3.12/site-packages/Py2001/
```

9.  Run the validation examples:

```
$ python validateP2001.py 

Processing file Validation_examples_ITU-R_P_2001_prof4_profile.csv
        Processing 1 / 2215, GHz=0.03, Tpc=0.001 % - 99.999 % ...
        Processing 444 / 2215, GHz=0.2, Tpc=0.001 % - 99.999 % ...
        Processing 887 / 2215, GHz=2.0, Tpc=0.001 % - 99.999 % ...
        Processing 1330 / 2215, GHz=20.0, Tpc=0.001 % - 99.999 % ...
        Processing 1773 / 2215, GHz=50.0, Tpc=0.001 % - 99.999 % ...

Processing file Validation_examples_ITU-R_P_2001_b2iseac_profile.csv
        Processing 1 / 2215, GHz=0.03, Tpc=0.001 % - 99.999 % ...
        Processing 444 / 2215, GHz=0.2, Tpc=0.001 % - 99.999 % ...
        Processing 887 / 2215, GHz=2.0, Tpc=0.001 % - 99.999 % ...
        Processing 1330 / 2215, GHz=20.0, Tpc=0.001 % - 99.999 % ...
        Processing 1773 / 2215, GHz=50.0, Tpc=0.001 % - 99.999 % ...
Validation results: 2 out of 2 tests passed successfully.
The deviation from the reference results is smaller than 1e-06.
```