#!/usr/bin/env python3
# import specific modules

import numpy as np
import os,sys, glob, shutil
import deap
import random, copy, time, pandas, math, operator, pickle
import tqdm
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .HGSPSO import *
from deap import base, benchmarks, creator, tools

class HgsTVAC(HgsPSO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def updateParticle(self, part): # {{{
        '''
        Update each particle's positions.
        '''

        random.seed(self.seed)

        best    = self.best
        limits  = self.slimits
        par     = self.par
        idx     = part.index

        if self.verbose:
            print(f' UPDATE PART: {idx}')
            print(f' Cur particle:{part}')
        old_part    = copy.deepcopy(part)
        old_pb      = copy.deepcopy(part.best)
        old_best    = copy.deepcopy(best)
        #LI part insert
        
        for i in par:
            if i == 'loc':
                c1 = self.loc_c1
                c2 = self.loc_c2
                iw = self.loc_w
            elif i == 'flux':
                c1 = self.flux_c1
                c2 = self.flux_c2
                iw = self.flux_w
            for j in part[i].index:
                target = part[i].loc[j]
                Gbest  = best[i].loc[j]
                Lbest  = part.best[i].loc[j]
                speed  = part[f'{i}_speed'].loc[j]

                max_val   = limits['max'][i].loc[j]
                max_speed = limits['max'][f'{i}_speed'].loc[j]
                min_val   = limits['min'][i].loc[j]
                min_speed = limits['min'][f'{i}_speed'].loc[j]

                u1 = np.random.uniform(0, c1, len(target))
                u2 = np.random.uniform(0, c2, len(target))

                v_u1 = [a * (b - c) for a, b, c in zip(u1, Lbest, target)]
                v_u2 = [a * (b - c) for a, b, c in zip(u2, Gbest, target)]

                sp = [ b + c for b, c in zip(v_u1, v_u2)]

                for sidx, ms in enumerate(max_speed):
                    if abs(sp[sidx]) > abs(ms):
                        sp[sidx] = math.copysign(ms, sp[sidx])

                part[f'{i}_speed'].loc[j] = sp

                target = target + sp

                for (tidx, maxs), mins in zip(enumerate(max_val), min_val):
                    if target[tidx] > maxs:
                        target[tidx] = maxs
                    elif target[tidx] < mins:
                        target[tidx] = mins

                part[i].loc[j] = target

        part.best = old_pb
        best      = old_best
        self.pop[idx] = part
        if self.verbose:
            print(f' pop updated: {self.pop[idx]}')
            print(f' part       : {part}') 
            # }}}

