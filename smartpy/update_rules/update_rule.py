#from ..misc.metaclasses import HyperparamsMeta
from ..misc import saveable, HasParams, HasHyperparams


@saveable
class UpdateRule(HasParams, HasHyperparams):
    def apply(self, gradients):
        raise NameError('Should be implemented by inheriting classes!')
