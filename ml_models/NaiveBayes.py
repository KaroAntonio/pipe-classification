from __future__ import division
from __future__ import absolute_import
from math import exp, pi

class NaiveBayes(object):    
    
    def __init__(self):
        self.labels = []
        self.stat_labels = {}
        self.numof_instances = 0
        self.attributes = []                
        self.attvals = {}
        self.real_stat = {}
        self.stat_attributes = {}
        self.smoothing = {}
        self.attribute_type = {}  
        self.gaussian = 1 / ((2 * pi) ** (1 / 2))
        
    def set_real(self, attributes):
        for attribute in attributes:
            self.attribute_type[attribute] = u'real' 
            
    def set_smoothing(self, attributes):
        for attribute in attributes.keys():
            self.smoothing[attribute] = attributes[attribute]    
    
    def drop_attributes(self, attributes):
        for attribute in attributes:
            del self.attvals[attribute]
            del self.stat_attributes[attribute]
            del self.attribute_type[attribute]
            if attribute in self.real_stat.keys():
                del self.real_stat[attribute]
            if attribute in self.smoothing.keys():
                del self.smoothing[attribute]        
        new_attributes = []
        for attribute in self.attributes:
            if attribute not in attributes:
                new_attributes.append(attribute)
        self.attributes = new_attributes        
        
    def add_instances(self, params):
        if not u'attributes' in params.keys() or not u'label' in params.keys() or not u'cases' in params.keys():
            raise Exception(u'Missing instance parameters')
        if len(self.stat_attributes.keys()) == 0:
            for attribute in params[u'attributes'].keys():
                self.stat_attributes[attribute] = {}
                self.attributes.append(attribute)
                self.attvals[attribute] = []
                if not attribute in self.attribute_type.keys():
                    self.attribute_type[attribute] = u'nominal'
        else:
            for attribute in self.attribute_type.keys():
                if not attribute in params[u'attributes'].keys():
                    raise Exception(u'Attribute not given in instance: ' + attribute)                    
        self.numof_instances += params[u'cases']
        if not params[u'label'] in self.stat_labels.keys():
            self.labels.append(params[u'label'])
            self.stat_labels[params[u'label']] = 0        
        self.stat_labels[params[u'label']] += params[u'cases']        
        for attribute in self.stat_attributes.keys():
            if not attribute in params[u'attributes'].keys():
                raise Exception(u'Attribute ' + attribute + u' not given')
            attval = params[u'attributes'][attribute]
            if not attval in self.stat_attributes[attribute].keys():
                self.attvals[attribute].append(attval)
                self.stat_attributes[attribute][attval] = {}
            if not params[u'label'] in self.stat_attributes[attribute][attval].keys():
                self.stat_attributes[attribute][attval][params[u'label']] = 0                
            self.stat_attributes[attribute][attval][params[u'label']] += params[u'cases']                           
                
    def train(self):
        self.model = {u'lprob' : {}, u'cprob' : {}, u'real_stat' : {}}        
        for label in self.stat_labels.keys():
            self.model[u'lprob'][label] = self.stat_labels[label] / self.numof_instances        
        for attribute in self.stat_attributes.keys():
            if not self.attribute_type[attribute] == u'real':
                self.model[u'cprob'][attribute] = {}
                for label in self.stat_labels.keys():
                    total = 0
                    attvals = []               
                    for attval in self.stat_attributes[attribute].keys():
                        if label in self.stat_attributes[attribute][attval].keys() and self.stat_attributes[attribute][attval][label] > 0:
                            attvals.append(attval)
                            if not attval in self.model[u'cprob'][attribute].keys():
                                self.model[u'cprob'][attribute][attval] = {}
                            self.model[u'cprob'][attribute][attval][label] = self.stat_attributes[attribute][attval][label]
                            total += self.model[u'cprob'][attribute][attval][label]
                    if attribute in self.smoothing.keys():
                        uc = self.smoothing[attribute]
                        if uc <= 0:
                            uc = 0.5
                        if not u'*' in self.model[u'cprob'][attribute].keys():
                            self.model[u'cprob'][attribute][u'*'] = {}
                        self.model[u'cprob'][attribute][u'*'][label] = uc;
                        total += uc
                        if u'*' in attvals:
                            raise Exception(u"'*' as attribute value has been reserved")
                        attvals.append(u'*')
                    for attval in attvals:
                        self.model[u'cprob'][attribute][attval][label] /= total
                        
            else:
                if attribute in self.smoothing.keys():
                    raise Exception(u'Smoothing has been set for real attribute ' + attribute)
                self.model[u'real_stat'][attribute] = {}
                for attval in self.stat_attributes[attribute].keys():
                    for label in self.stat_attributes[attribute][attval]:
                        if not label in self.model[u'real_stat'][attribute]:
                            self.model[u'real_stat'][attribute][label] = {u'sum' : 0, u'count' : 0, u'mean' : 0, u'sigma' : 0}
                        self.model[u'real_stat'][attribute][label][u'sum'] += float(attval) * self.stat_attributes[attribute][attval][label]
                        self.model[u'real_stat'][attribute][label][u'count'] += float(self.stat_attributes[attribute][attval][label])
                        if self.model[u'real_stat'][attribute][label][u'count'] > 0:
                            self.model[u'real_stat'][attribute][label][u'mean'] = self.model[u'real_stat'][attribute][label][u'sum'] / self.model[u'real_stat'][attribute][label][u'count']
                for attval in self.stat_attributes[attribute].keys():
                    for label in self.stat_attributes[attribute][attval]:
                        self.model[u'real_stat'][attribute][label][u'sigma'] += (float(attval) - self.model[u'real_stat'][attribute][label][u'mean']) ** 2 * self.stat_attributes[attribute][attval][label]           
                for label in self.model[u'real_stat'][attribute]:
                    self.model[u'real_stat'][attribute][label][u'sigma'] = (self.model[u'real_stat'][attribute][label][u'sigma'] / (self.model[u'real_stat'][attribute][label][u'count'] - 1)) ** (1 / 2)       
                    
    def predict(self, params):
        if not u'attributes' in params.keys():
            raise Exception(u'Missing attributes parameter')
        scores = {}
        nsum = 0.0
        nscores = {}
        for label in self.labels:
            scores[label] = self.model[u'lprob'][label]
        for attribute in params[u'attributes'].keys():
            if not attribute in self.attribute_type:
                raise Exception(u'Unknown attribute ' + attribute)
            if not self.attribute_type[attribute] == u'real':
                attval = params[u'attributes'][attribute]
                if not attval in self.stat_attributes[attribute] and not attribute in self.smoothing.keys():
                    raise Exception(u'Attribute value ' + attval + u' not defined')
                for label in self.labels:
                    if attval in self.model[u'cprob'][attribute] and label in self.model[u'cprob'][attribute][attval] and self.model[u'cprob'][attribute][attval][label] > 0:
                        scores[label] *= float(self.model[u'cprob'][attribute][attval][label])
                    elif attribute in self.smoothing.keys():
                        scores[label] *= float(self.model[u'cprob'][attribute][u'*'][label])
                    else:
                        scores[label] = 0
            else:
                for label in self.labels:
                    nscores[label] = float(self.gaussian / self.model[u'real_stat'][attribute][label][u'sigma'] * exp(-0.5 * ((float(params[u'attributes'][attribute]) - self.model[u'real_stat'][attribute][label][u'mean']) / self.model[u'real_stat'][attribute][label][u'sigma']) ** 2))
                    nsum += float(nscores[label])
                if not nsum == 0:
                    for label in self.labels:
                        scores[label] *= nscores[label]       
        sumPx = float(0)
        for label in scores.keys():
            sumPx += float(scores[label])
        for label in scores.keys():
            scores[label] /= float(sumPx+0.000001)
        return(scores)

    def preferredLabel(self, scores):
        maxValue = None
        prefLabel = None
        for label in scores.keys():
            if maxValue == None:
                maxValue = scores[label]
                prefLabel = label
            elif scores[label] > maxValue:
                maxValue = scores[label]
                prefLabel = label
        return(prefLabel)


