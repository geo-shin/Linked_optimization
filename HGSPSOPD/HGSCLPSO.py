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

class HgsCLPSO(HgsPSO):  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Pc(self): # {{{
        # Learning Probability Pc
        t = np.linspace(0, 5, self.npop)
        for i in range(self.npop):
            self.pc[i] = 0 + 0.5*(np.exp(t[i]) - np.exp(t[0])) / (np.exp(5) - np.exp(t[0]))
            if len(self.f_pbest) < self.npop:
                self.f_pbest.append([i]*self.D)
            else:
                pass
        if self.verbose:
            print(self.pc) # }}}

    def ComprehensiveLearning(self): # {{{
        f_pbest = self.f_pbest
        if self.fi1 is None:
            self.fi1 = np.zeros(self.D)
        if self.fi2 is None:
            self.fi2 = np.zeros(self.D)
        if self.fi is None:
            self.fi  = np.zeros(self.D)
        if self.bi is None:
            self.bi  = np.zeros(self.D)
        # Check Learning status
        for v in range(self.npop):
            if self.mchange[v] > self.minC: # 나중에 부등호 변경
                self.mchange[v] = int(0)
                for z in range(self.D):
                    self.fi1[z] = math.ceil(random.uniform(0,1)*(self.npop-1))
                    self.fi2[z] = math.ceil(random.uniform(0,1)*(self.npop-1))
                    self.fi1 = self.fi1.astype(int)
                    self.fi2 = self.fi2.astype(int)
                    if self.verbose:
                        print(f"fi1[{z}]: {self.fi1[z]}, fi2[{z}]: {self.fi2[z]}")
                        print(f" fitness np : {self.fit_best}")
                    self.fi[z] = np.where(self.fit_best[self.fi1[z]] < self.fit_best[self.fi2[z]], self.fi1[z], self.fi2[z]).tolist()
                    if self.verbose:
                        print(f"self.fi  = {self.fi}")
                    self.bi1 = random.random() -1 + self.pc[v]
                    self.bi[z] = np.where(self.bi1 >= 0, 1, 0).tolist()
                    if self.verbose:
                        print(f"self.bi1 = {self.bi1}")
                        print(f"self.bi  = {self.bi}")
                if np.sum(self.bi) == 0:
                    rc = round(random.uniform(0,1)*(self.D -1))
                    if self.verbose:
                        print(f"rc: {rc}")
                    rc = int(rc)
                    self.bi[rc] = 1

                for m in range(self.D):
                    f_pbest[v][m] = int(self.bi[m]*self.fi[m] + (1 - self.bi[m])*f_pbest[v][m])
                    if self.verbose:
                        print('self.bi[m]:', self.bi[m])
                        print('f_pbest[v][m]:', f_pbest[v][m])
                        print('f_pbest:', f_pbest)
        if self.verbose:
            print('Comprensive_learning: end')
        return f_pbest # }}}
    
    def fbest(self): # {{{

        self.CLbest = copy.deepcopy(self.pop)
        d = np.zeros([self.npop,0])
        keys  = []

        for j in self.par:
            key = self.pop[0][j].keys()
            lloc = np.zeros([self.npop, len(key)])
            for s in self.source:
                for i in range(self.npop):
                    lloc[i] = self.pop[i].best[j].loc[s]
                d = np.hstack((d, lloc))
                keys.append(len(key))
        d_1 = np.copy(d)
        for j in range(0, self.npop):
            for k in range(0, self.D):
                index_1 = self.f_pbest[j][k]
                d_1[j,k] = d[index_1,k]

        sidx = 0
        data = []
        for size in keys:
            eidx = sidx + size
            data.append(d_1[:, sidx:eidx])
            sidx = eidx

        if self.verbose:
            print(f'data for CLPSO: {data}')

        # setting fbest
        idx = 0
        for j in self.par:
            for s in self.source:
                for i in range(self.npop):
                    self.CLbest[i][j].loc[s] = data[idx][i]
                idx += 1

        if self.verbose:
            print(f'CLPSO: {self.CLbest[0]}')
            print(f'pop  : {self.pop[0].best}')
            print(f'pop  : {self.pop[0]}') # }}}

    def updateParticle(self, part): # {{{
        '''
        Update each particle's positions.
        '''
        random.seed(self.seed)

        best    = self.best
        limits  = self.slimits
        par     = self.par
        idx     = part.index

        # CLPSO
        if self.g is not self.CLg:
            if self.verbose:
                gtext = '   CLPSO PART'
                self.declare(gtext)

            HgsCLPSO.Pc(self)
            self.f_pbest = HgsCLPSO.ComprehensiveLearning(self)
            HgsCLPSO.fbest(self)
            self.CLg = self.g
        else:
            pass

        if self.verbose:
            print(f' UPDATE PART: {idx}')
            print(f' Cur particle:{part}')

        old_part    = copy.deepcopy(part)
        old_pb      = copy.deepcopy(part.best)
        old_best    = copy.deepcopy(best)

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
                Cbest  = self.CLbest[idx][i].loc[j]
                speed  = part[f'{i}_speed'].loc[j]

                max_val   = limits['max'][i].loc[j]
                max_speed = limits['max'][f'{i}_speed'].loc[j]
                min_val   = limits['min'][i].loc[j]
                min_speed = limits['min'][f'{i}_speed'].loc[j]

                u1 = np.random.uniform(0, c1, len(target))
                u2 = np.random.uniform(0, c2, len(target))

                v_u1 = [a * (b - c) for a, b, c in zip(u1, Cbest, target)]
                v_u2 = [a * (b - c) for a, b, c in zip(u2, Gbest, target)]

                sp = [(iw * a) + b + c for a, b, c in zip(speed, v_u1, v_u2)]

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
