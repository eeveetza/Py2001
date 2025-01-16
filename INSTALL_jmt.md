# Install instructions to get Py2001 running on Ubuntu/Win10 machine with VS Code

1.  `git clone` this repo (a fork of [https://github.com/eeveetza/Py2001]):

~~~
    $ git clone https://github.com/joshuamhtsang/Py2001.git
~~~

2.  Open VS Code in the repo directory:  

~~~
    $ code .
~~~

3.  In Command Palette (usually crtl + shift + P), install a new venv.

4.  Open a terminal in VS Code and ensure it's running the new venv.  If not, `source` it.

5.  In the terminal, run (as explained in the `README.md`):

~~~
    $ python -m pip install "git+https://github.com/eeveetza/Py2001/#egg=Py2001" `
~~~

6.  