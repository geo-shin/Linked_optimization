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

class HgsFDR(HgsPSO):  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def FDR(self): # {{{
        # step 0: orgarnize particle best & fitness values and regulization
        fval0 = np.zeros(self.npop)
        d     = np.zeros([self.npop,0])
        keys  = []
        mus   = []
        stds  = []
        for j in self.par:
            key = self.pop[0].best[j].keys()
            lloc = np.zeros([self.npop, len(key)])
            for s in self.source:
                for i in range(self.npop):
                    lloc[i] = self.pop[i].best[j].loc[s]
                    fval0[i] = self.pop[i].best.fitness.values[0]
                mu  = np.mean(lloc, axis=0)
                #print(f'mu : {mu}')
                std = np.std(lloc, axis=0)
                #print(f'std: {std}')
                for i, istd in enumerate(std):
                    if istd == 0:
                        std[i] = 1.0e-10
                nloc = (lloc - mu) / std
                #print(f'nloc : {nloc}')
                d = np.hstack((d, nloc))
                keys.append(len(key))
                mus.append(mu)
                stds.append(std)
        fmu = np.mean(fval0)
        fstd = np.std(fval0)
        fval = (fval0 - fmu) / fstd
        D = len(d[0])
        Pnd = np.zeros_like(d)
        # 거리계산
        for k in range(self.npop):
            dis = np.abs(d[k, :] - d)
            fiterr = fval[k] - fval[:, None]
            fiterr = np.tile(fiterr, (1,D))
            fiterr -= (dis == 0) * fiterr
            dis += (dis == 0)
            FDR = fiterr / dis
            for dimcnt in range(D):
                Fid = np.argmax(FDR[:, dimcnt])
                Pnd[k, dimcnt] = d[Fid, dimcnt]

        sidx = 0
        data = []
        for size in keys:
            eidx = sidx + size
            data.append(Pnd[:, sidx:eidx])
            sidx = eidx
        for i in range(len(data)):
            data[i] = data[i] * stds[i] + mus[i]
        if self.verbose:
            print(f'data for FDR: {data}')

        # setting FDRbest for update
        self.FDRbest = copy.deepcopy(self.pop)
        idx = 0
        for j in self.par:
           for s in self.source:
               for i in range(self.npop): 
                   self.FDRbest[i][j].loc[s] = data[idx][i]
               idx += 1
        if self.verbose:
            print(f'FDR: {self.FDRbest[0]}')
            print(f'pop: {self.pop[0]}') # }}}

    def updateParticle(self, part): # {{{
        '''
        Update each particle's positions.
        '''
        random.seed(self.seed)

        c1 = self.c1
        c2 = self.c2

        best    = self.best
        limits  = self.slimits
        par     = self.par
        idx     = part.index
        # FDR cal
        if self.g is not self.FDRg:
            if self.verbose:
                gtext = '   FDR PART'
                self.declare(gtext)
            HgsFDR.FDR(self)
            self.FDRg = self.g
        else:
            pass

        if self.verbose:
            print(f' UPDATE PART: {idx}')
            print(f' Cur particle:{part}')

        old_part    = copy.deepcopy(part)
        old_pb      = copy.deepcopy(part.best)
        old_best    = copy.deepcopy(best)

        for i in par:
            for j in part[i].index:
                target = part[i].loc[j]
                Gbest  = best[i].loc[j]
                Lbest  = part.best[i].loc[j]
                Fbest  = self.FDRbest[idx][i].loc[j]
                speed  = part[f'{i}_speed'].loc[j]

                max_val   = limits['max'][i].loc[j]
                max_speed = limits['max'][f'{i}_speed'].loc[j]
                min_val   = limits['min'][i].loc[j]
                min_speed = limits['min'][f'{i}_speed'].loc[j]

                u1 = np.random.uniform(0, c1, len(target))
                u2 = np.random.uniform(0, c2, len(target))

                v_u1 = [a * (b - c) for a, b, c in zip(u1, Lbest, target)]
                v_u2 = [a * (b - c) for a, b, c in zip(u2, Gbest, target)]
                v_u3 = [a * (b - c) for a, b, c in zip(u2, Fbest, target)]

                sp = [(self.w * a) + b + c + d for a, b, c, d in zip(speed, v_u1, v_u2, v_u3)]

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
        self.best = old_best
        self.pop[idx] = part
        if self.verbose:
            print(f' pop updated: {self.pop[idx]}') 
            print(f' part       : {part}')
            # }}}


