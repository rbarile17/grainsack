Extension
=========

Add new LP-X methods.

.. code-block:: python

    def method_1(prediction):
        pass

    def method_2(prediction):
        pass

    def explain_factory(lpx_config):
        method = lpx_config("method")
        if method == "method_1":
            return method_1
        else:
            return method_2

    luigi.build([Comparison("explain_factory")])

Two custom LP-X methods (`method_1`, `method_2`) are declared as functions.
Additionally, a function factory (`explain_factory`) is used to return the appropriate function based on the explanation config.
