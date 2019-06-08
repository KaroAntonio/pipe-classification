from fuzzylogic import *

def criticality_model( pd, es, ac ):
    '''
    pd: Pipe Diameter
    es: Environmentally Sensitive
    ac: Accessibility
    '''
    pd_bounds = {
            0:[0,0,300,500],
            10:[300,400,750,850],
            15:[650,750,float('inf')],
            }

    es_bounds = {
            0:[0,0,10],
            15:[6,15,float('inf')],
            }

    ac_bounds = {
            0:[0,0,6],
            10:[4,8,12],
            15:[10,16, float('inf')],
            }

    out_bounds = {
            2: [0,1.75,3.25],
            3: [1.75,3.25,5],
            4: [3.25,5,6.5],
            5: [5,6.5,8],
            6: [6.5,8,float('inf')],
            }

    pd_out = membership( pd, pd_bounds )
    es_out = membership( es, es_bounds )
    ac_out = membership( ac, ac_bounds )

    out = sum([pd_out, es_out, ac_out]) * 2./9 # 1/6 * 4/3

    return membership( out, out_bounds )

def condition_model(rl, tb, rb, mi ):
    '''
    rl: Remaining Life
    tb: Total Breaks
    rb: Recent Number of Breaks
    mi: Maintenance Index
    return an integer condition rating
    '''

    # NOTE what are the mi bounds?
    # desc is in range 0 - 5%, bounds is in (0,0.06)...?
    # should the bounds at the top end always go to infinity ...?

    rl_bounds = {
            0: [45,53,54, float('inf')],
            5: [25,33,48,55],
            15: [10,17,28,35],
            20: [0,0,12,20]
            }

    tb_bounds = {
            0: [0,0,1,2],
            5: [0,2,4.5, 6],
            15: [4,5.5,8,10],
            20: [8,9,10,float('inf')]
            }

    rb_bounds = {
            0: [0,0,0.5,1],
            5: [0.5,1,2.5,3],
            15: [2.5,3,4.5,5],
            20: [4.5,5,6,float('inf')]
            }

    mi_bounds = {
            5: [0,0,0.01,0.02],
            10: [0.01,0.02,0.04,0.05],
            15: [0.04,0.045,0.06,float('inf')]
            }

    out_bounds = {
            1: [0,0,1.75],
            2: [0,1.75,3.25],
            3: [1.75,3.25,5],
            4: [3.25,5,6.5],
            5: [5,6.5,8],
            6: [6.5,8,10],
            7: [8,10,float('inf')]
            }

    rl_out = membership( rl, rl_bounds )
    tb_out = membership( tb, tb_bounds )
    rb_out = membership( rb, rb_bounds )
    mi_out = membership( mi, mi_bounds )

    # print( rl_out, tb_out, rb_out, mi_out )

    # Input to the condition model bound is the sum of the prior outputs
    out = sum([rl_out, tb_out, rb_out, mi_out]) * 1./6

    return membership( out, out_bounds )

def performance_model( hc, qy, cs, pd, pt, land_use_type ):
    '''
    pt: Pipe Type
    hc: Hydraulic Capacity
    qy: Quality
    cs: Conformance to Standard
    pd: Pipe Diameter (mm)
    return an integer performance grade
    '''
    if pd >= 600:
        hc_bounds = {
                0:[0,0,1,1.75],
                15:[1,1.5,2.25,2.75],
                30:[2,2.5,3.5,float('inf')]
                }
    else:
        hc_bounds = {
                0:[0,0,1,2.25],
                15:[1,2,4,5.5],
                30:[4,5,6,float('inf')]
                }


    qy_bounds = {
            0:[0,0,0.5],
            15:[14.5,15,15.5]
            }

    inv_land_use_cats = [
                'Industry',
                'Schools'
            ]
    if pd < 19: cs_out = 10
    elif pd < 100 and pt == "Copper": cs_out = 15
    elif pd < 300 and land_use_type in inv_land_use_cats : cs_out = 15
    else: cs_out = 0

    out_bounds = {
            1: [0,0,1.75],
            2: [0,1.75,3.25],
            3: [1.75,3.25,5],
            4: [3.25,5,6.5],
            5: [5,6.5,8],
            6: [6.5,8,10],
            7: [8,10,float('inf')]
            }

    hc_out = membership( hc, hc_bounds )
    qy_out = membership( qy, qy_bounds )
    cs_out = membership( cs, cs_bounds )

    out = sum([hc_out, qy_out, cs_out]) * 2./9 # 1/6 * 4/3

    return membership( out, out_bounds )

