from utils import *

def get_condition_params():
    return {
            'model_name':   'condition_model',
            'model':        'condition_model',
            'var_map_fid':      'var_maps/condition_var_map.json',
            #'out_f':       condition_calc_out,
            'out_f':        'condition_exp_out',
            'val_map_fid':      'val_maps/condition_exp_map.json',
            'params_fid':       'models/condition_model_opt.csv'
            }

def get_criticality_params():
    return {
            'model_name':   'criticality_model',
            'model':        'criticality_model',
            'var_map_fid':      'var_maps/criticality_var_map.json',
            'out_f':        'criticality_exp_out',
            'params_fid':       'models/criticality_model_opt.csv'
            }

def get_performance_params():
    return {
            'model_name':   'performance_model',
            'model':        'performance_model',
            'var_map_fid':      'var_maps/performance_var_map.json',
            'out_f':        'performance_exp_out',
            'params_fid':       'models/performance_model_opt.csv'
            }

def get_mitigation_params():
    return {
		'model_name':   'mitigation_model',
		'var_map_fid':   'var_maps/mitigation_var_map.json',
            }

def prep_params( p ):
	if 'var_map_fid' in p and 'var_map' not in p:
		p['var_map'] = json.load(open(p['var_map_fid'], 'r'))	

	p['out'] = p['var_map']['out']
	return p
