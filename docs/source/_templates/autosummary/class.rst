{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
{% if 'LikelihoodModel' in objname %}
   :special-members: __call__
{% endif %}
