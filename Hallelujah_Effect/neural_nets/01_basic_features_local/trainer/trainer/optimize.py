import shutil

from .model import train_and_evaluate
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.callbacks import VerboseCallback


class CustomCallback(object):
    def __init__(self, base_output_dir):
        self.base_ouput_dir = base_output_dir
        self.no_iters = 1

    def __call__(self, res):
        self.no_iters += 1

    def get_output_dir(self):
        import os
        return os.path.join(self.base_ouput_dir, str(self.no_iters))


class BetterVerboseCallback(VerboseCallback):
    def __init__(self):
        super().__init__(0)

    def __call__(self, res):
        super().__call__(res)

        if self.iter_no == self.n_total:
            print('Results on last run:')
            print(res)


def optimize(arguments):
    tuning_parameters = parse_config()
    trials = tuning_parameters['max_trials']
    space = tuning_parameters['space']

    shutil.rmtree(arguments['output_dir'], ignore_errors=True)

    output_dir_callback = CustomCallback(arguments['output_dir'])
    callbacks = [output_dir_callback, BetterVerboseCallback()]

    @use_named_args(space)
    def wrapped_train_and_evaluate(**args):
        args['output_dir'] = output_dir_callback.get_output_dir()
        merged_args = {**arguments, **args}

        result = train_and_evaluate(merged_args)

        eps = 10 ** -10

        beta = 0.5

        positive_support = result['true_positives'] + result['false_negatives']
        negative_support = result['true_negatives'] + result['false_positives']

        positive_recall = result['true_positives'] / (1. * positive_support + eps)
        positive_precision = result['true_positives'] / (1. * (result['true_positives'] + result['false_positives']) + eps)
        f1_positive = (1. + beta) * (
                    (positive_recall * positive_precision) / ((beta * positive_precision) + positive_recall + eps))

        negative_recall = result['true_negatives'] / (1. * negative_support + eps)
        negative_precision = result['true_negatives'] / (1. * (result['true_negatives'] + result['false_negatives']) + eps)
        f1_negative = (1. + beta) * (
                    (negative_recall * negative_precision) / ((beta * negative_precision) + negative_recall + eps))

        f1_weighted = (f1_positive * (1. * positive_support) / (positive_support + negative_support + eps)) + \
                      (f1_negative * (1. * negative_support) / (positive_support + negative_support + eps))
        result['f1_weighted'] = f1_weighted

        print('***** result *****')
        print(result)

        if tuning_parameters['goal'] == 'MAXIMIZE':
            return -1. * result[tuning_parameters['metric']]
        else:
            return result[tuning_parameters['metric']]

    # TODO: Unify random state
    res = gp_minimize(wrapped_train_and_evaluate,
                      space,
                      n_calls=trials,
                      random_state=42,
                      callback=callbacks)
    print(res)


def parse_config():
    import yaml

    with open('hyperparam.yaml', 'r') as stream:
        raw_params = yaml.load(stream)

    goal = raw_params['goal']
    max_trials = raw_params['maxTrials']
    metric = raw_params['hyperparameterMetricTag']
    space = [parse_param(p) for p in raw_params['params']]

    return {'goal': goal, 'max_trials': max_trials, 'metric': metric, 'space': space}


def parse_param(param):
    p = None

    if param['type'] == 'INTEGER':
        p = Integer(
            param['minValue'],
            param['maxValue'],
            # transform=param['scaleType'],
            name=param['parameterName']
        )
    if param['type'] == 'REAL':
        p = Real(
            param['minValue'],
            param['maxValue'],
            prior=param['scaleType'],
            name=param['parameterName']
        )
    if param['type'] == 'CATEGORICAL':
        p = Categorical(
            param['categoricalValues'],
            name=param['parameterName']
        )

    return p
