.. _features-label:


==========================
The FeaturesComputer class
==========================

FeaturesComputer is the main class used to extract features from a signal.

The class defines two class variables:

* ``EXPORT``: (set) it keeps track of the names of the features to be exported
* ``FEATFUNCS``: (dict) it keeps track of functions associated to each feature name

For each block of data, all features defined in ``FEATFUNCS`` are calculated,
but only the subset defined in ``EXPORT`` is returned by the compute() method.

The class defines two decorators used to register features :

* ``globalfeature``: this registers a feature for computation **and** for export.
* ``globalnoexport``: this registers a feature for computations only.

.. note::

    In addition, we also define two functions ``feature()`` and ``noexport()`` that wrap the above decorators.

Features registration
---------------------

The package ``__init__.py`` file has the following import statements:

.. code-block:: python

	from wavely.signal.features.features import FeaturesComputer
	from wavely.signal.features.spectralfeatures import *
	from wavely.signal.features.acousticfeatures import *

What happens is that first, ``FeaturesComputer`` is loaded. Then all the functions
in ``spectralfeatures.py`` and ``acousticfeatures.py`` are loaded. Every time we encounter a function
with a ``@feature`` or ``@noexport`` decorator, it is loaded in the appropriate FeaturesComputer class variables (``FEATFUNCS`` / ``EXPORT``) and thereby registered for computation.

Features computation
--------------------

The FeaturesComputer.compute() method works as follows :

* All computed features are held in the dictionnary ``self.feats``.
* ``self.feats`` is first initialised with the parameters 'rate', 'nfft' and 'window' (known since they are passed to the class __init__ method and stored in ``self.params``)
* for each feature in ``FEATFUNCS``, the feature is computed and added to ``self.feats``.
* special case: the feature 'prevnormspectrum' is obtained from the 'normspectrum' computed from the previous batch.
* when all features are computed, the features registered in ``EXPORT`` are returned.

.. important::

    **Arguments of feature functions must be features themselves !**.

	For example, assuming we have a registered feature ``my_feature(x, y)``,
	then ``x`` and ``y`` must either be feature functions, or set in the FeaturesComputer class initial parameters
	(such as the rate, nfft, etc.).
	Indeed, when computing the feature ``my_feature``, the FeaturesComputer class will match its argument and look
	for ``x`` and ``y`` in ``FEATFUNCS``, so they must be registered too. The main point here is to cache the
	intermediate results, for example if we have another feature function ``my_feature2(x, z)``,
	the following will happen when calling the compute() method:

	* try to compute ``my_feature``, since it has two arguments we have to compute them first
	* compute ``x`` and store it in self.feats['x']
	* compute ``y`` and store it in self.feats['y']
	* compute ``my_feature`` and store it in self.feats['my_feature'] (using the previously computed values of x and y)
	* try to compute ``my_feature2``, since it has two arguments we have to compute them first
	* ``x`` was already calculated and is available in self.feats['x'], so we skip it
	* compute ``z`` and store it in self.feats['z']
	* compute ``my_feature2`` and store it in self.feats['my_feature2'] (using the previously computed values of x and z)
