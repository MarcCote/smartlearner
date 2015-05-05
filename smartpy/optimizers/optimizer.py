import theano
import theano.tensor as T


class Optimizer(object):
    def __init__(self, model, loss_fct, dataset, update_rules=[], param_modifiers=[]):
        self.update_rules = update_rules
        self.param_modifiers = param_modifiers

        # Build learner
        nb_datasets = 1
        self.inputs = [T.matrix('input' + str(i)) for i in range(nb_datasets)]
        self.loss = loss_fct(*self.inputs)

        self.gradients, updates = model.get_gradients(self.objective)
        self.updates.update(updates)

        # Apply update rules
        for update_rule in self.update_rules:
            gradients, updates = update_rule.apply(self.gradients)
            self.gradients.update(gradients)
            self.updates.update(updates)  # Add updates from update_rule

        # Update parameters
        for param, gparam in self.gradients.items():
            self.updates[param] = param - self.gradients[param]

        # Modify parameters, if needed
        # for param_modifier in self.param_modifiers:
        #     for param, gparam in self.gradients.items():
        #     gradients, updates = param_modifier.apply(self.gradients)
        #     self.gradients.update(gradients)
        #     self.updates.update(updates)  # Add updates from update_rule


    def save(self, savedir="./"):
        for update_rule in self.update_rules:
            update_rule.save(savedir, update_rule.__class__.__name__)

    def load(self, loaddir="./"):
        for update_rule in self.update_rules:
            update_rule.load(loaddir, update_rule.__class__.__name__)

