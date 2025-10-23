============
Installation
============

Pre-requisites
--------------

Python 3.5 and virtualenv

For development
---------------

* Clone from github::

    git clone git@github.com:Wavely/signal.git
    cd signal/

* Create a virtual environment and install the dev requirements::

    virtualenv -p python3.5 ~/.virtualenv/signal
    source ~/.virtualenv/signal/bin/activate
    pip install -U pip setuptools
    pip install -e .\[dev\]

* Run the tests::

    python -m unittest

For use in another project
--------------------------

Upgrade pip and setuptools::

    pip install -U pip setuptools


Install from github using pip::

    pip install git+ssh://git@github.com/Wavely/signal.git

You an also install a specific commit, branch or tag, for example::

    pip install git+ssh://git@github.com/Wavely/signal.git@v0.0.2
