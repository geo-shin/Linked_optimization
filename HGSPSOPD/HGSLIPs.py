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

class HgsLIPs(HgsPSO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def LI(self): # {{{
        # step 0: orgarnize particle best & fitness values and regulization
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
        sidx = 0
        data = []
        nsize = 3
        if self.fai is not None:
            pass
        else:
            self.fai = np.zeros((self.npop,nsize,self.D))

        for i in range(self.npop):
            dist = np.linalg.norm(d[i] - d, axis=1)
            dist[i] = np.inf
            nearidx = np.argsort(dist)[:nsize]
            fai = (4.1 / nsize) * np.random.rand(nsize, self.D)
            self.fai[i] = fai
            if self.verbose:
                print(f'fai shape: {self.fai.shape}')
                print('fai', fai)
                print(f'd[nearidx] : {d[nearidx]}')
            dbest = np.sum(fai * d[nearidx], axis=0) / np.sum(fai)
            if self.verbose:
                print(f' dbest: {dbest}')
        for size in keys:
            eidx = sidx + size
            data.append(d[:, sidx:eidx])
            sidx = eidx
        for i in range(len(data)):
            data[i] = data[i] * stds[i] + mus[i]

        # setting LIbest for update
        self.LIbest = copy.deepcopy(self.pop)
        idx = 0
        for j in self.par:
           for s in self.source:
               for i in range(self.npop): 
                   self.LIbest[i][j].loc[s] = data[idx][i]
               idx += 1
        if self.verbose:
            print(f'LI : {self.LIbest[0]}')
            print(f'pop: {self.pop[0].best}') # }}}

    def updateParticle(self, part): # {{{
        '''
        Update each particle's positions.
        '''
        random.seed(self.seed)

        best    = self.best
        limits  = self.slimits
        par     = self.par
        idx     = part.index

        if self.g is not self.LIg:
            if self.verbose:
                gtext = '   LIPS PART'
                self.declare(gtext)
            HgsLIPs.LI(self)
            self.LIg = self.g
        else:
            pass

        fai = np.sum(self.fai[idx])
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
                Lbest  = self.LIbest[idx][i].loc[j]
                speed  = part[f'{i}_speed'].loc[j]

                max_val   = limits['max'][i].loc[j]
                max_speed = limits['max'][f'{i}_speed'].loc[j]
                min_val   = limits['min'][i].loc[j]
                min_speed = limits['min'][f'{i}_speed'].loc[j]

                u1 = np.random.uniform(0, c1, len(target))
                u2 = np.random.uniform(0, c2, len(target))

                v_u1 = [fai*(b - c) for b, c in zip(Lbest, target)]

                sp = [0.7298 * (a + b) for a, b in zip(speed, v_u1)]

                for sidx, ms in enumerate(max_speed):
                    if abs(sp[sidx]) > abs(ms):
                        sp[sidx] = math.copysign(ms, sp[sidx])

                part[f'{i}_speed'].loc[j] = sp

                target = target + sp
                for (tidx, maxs), mins in zip(enumerate(max_val), min_val):
                    if target[tidx] > maxs:
                        if random.random() < self.bprob:
                            target[tidx] = maxs
                        else:
                            target[tidx] = random.uniform(mins, maxs)

                    elif target[tidx] < mins:
                        if random.random() < self.bprob:
                            target[tidx] = mins
                        else:
                            target[tidx] = random.uniform(mins, maxs)
                part[i].loc[j] = target

            part.best = old_pb
            best      = old_best

        self.pop[idx] = part

        if self.verbose:
            print(f' pop updated: {self.pop[idx]}')
            print(f' part       : {part}')  
            # }}}
