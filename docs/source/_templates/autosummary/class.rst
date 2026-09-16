{{ objname | escape | underline }}

``{{ fullname }}``

{% if fullname in member_lookup_classes %}
See :doc:`methods and properties <../{{ fullname }}.members>` for individual
lookup pages, or the :doc:`alphabetical API dictionary <../all>` to search
all classes. The full class contract and existing member anchors are retained below.
{% endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
{% if 'LikelihoodModel' in objname %}
   :special-members: __call__
{% endif %}
