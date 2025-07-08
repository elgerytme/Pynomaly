Taxonomy Reference
==================

This section contains detailed information about Pynomaly's anomaly classification taxonomy.

.. toctree::
   :maxdepth: 2
   :caption: Taxonomy:

   anomaly-classification-taxonomy
   severity-classification
   type-classification

Classification Overview
-----------------------

Pynomaly implements a comprehensive two-dimensional anomaly classification system:

* **Severity Classification**: Categorizes anomalies by business impact and urgency
* **Type Classification**: Categorizes anomalies by structural characteristics

For complete documentation, see :doc:`anomaly-classification-taxonomy`.

API Documentation
-----------------

.. automodule:: pynomaly.application.services.anomaly_classification_service
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynomaly.domain.services.anomaly_classifiers.SeverityClassifier
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynomaly.domain.services.anomaly_classifiers.TypeClassifier
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynomaly.domain.services.anomaly_classifiers.DefaultSeverityClassifier
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynomaly.domain.services.anomaly_classifiers.DefaultTypeClassifier
   :members:
   :undoc-members:
   :show-inheritance:
