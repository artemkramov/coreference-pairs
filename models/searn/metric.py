import numpy


class Metric:

    name = ''

    def evaluate(self, state1, state2):
        pass


class BCubed(Metric):

    name = 'b-cubed'

    def evaluate(self, state1, state2):
        cdict = self.state_to_dict(state1)
        ldict = self.state_to_dict(state2)
        pass

    @staticmethod
    def state_to_dict(state: 'State'):
        ldict = {}
        for idx, mention in enumerate(state.mentions):
            if not (mention.cluster_id in ldict):
                ldict[mention.cluster_id] = []
            ldict[mention.cluster_id].append(idx)
        return ldict


    @staticmethod
    def mult_precision(el1, el2, cdict, ldict):
        """Computes the multiplicity precision for two elements."""
        return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
               / float(len(cdict[el1] & cdict[el2]))

    @staticmethod
    def mult_recall(el1, el2, cdict, ldict):
        """Computes the multiplicity recall for two elements."""
        return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
               / float(len(ldict[el1] & ldict[el2]))

    def precision(self, cdict, ldict):
        """Computes overall extended BCubed precision for the C and L dicts.
        Parameters
        ==========
        cdict: dict(item: set(cluster-ids))
            The cluster assignments to be evaluated
        ldict: dict(item: set(cluster-ids))
            The ground truth clustering
        """
        return numpy.mean([numpy.mean([self.mult_precision(el1, el2, cdict, ldict) \
                                       for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])

    def recall(self, cdict, ldict):
        """Computes overall extended BCubed recall for the C and L dicts.
        Parameters
        ==========
        cdict: dict(item: set(cluster-ids))
            The cluster assignments to be evaluated
        ldict: dict(item: set(cluster-ids))
            The ground truth clustering
        """
        return numpy.mean([numpy.mean([self.mult_recall(el1, el2, cdict, ldict) \
                                       for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])

    def f_score(self, cdict, ldict, beta=1.0) -> float:
        p_val = self.precision(cdict, ldict)
        r_val = self.recall(cdict, ldict)
        return (1.0 + beta ** 2) * (p_val * r_val / (beta ** 2 * p_val + r_val))
