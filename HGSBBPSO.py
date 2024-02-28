#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap 
import random, copy, time, pandas, math, operator, pickle
import tqdm
from .HGSPSO import *
from deap import base, benchmarks, creator, tools

class HgsBBPSO(HgsPSO):
    def __init__(self, prefix, sources_limits, time_list, obs,
            seed:int       =45,
            model_dir:str  ='base_model',
            tem_path:str   ='tem_path',
            generation:int =1,
            npop:int       =100,
            verbose:bool   =True, 
            pso_name:str   ='pso',
            nlimit:int     = 20,  
            phi1:int       = 1.5,
            phi2:int       = 1.5,
            grid_size:int  = 100,
            chunk_size:int = 10,
            exe_grok:str   ='grok',
            exe_hgs:str    ='hgs',
            ):
        super().__init__(prefix, sources_limits, time_list, obs, 
                         seed       = seed,
                         model_dir  = model_dir,
                         tem_path   = tem_path,
                         generation = generation,
                         npop       = npop,
                         verbose    = verbose,
                         pso_name   = pso_name,
                         nlimit     = nlimit,
                         phi1       = phi1,
                         phi2       = phi2,
                         grid_size  = grid_size,
                         chunk_size = chunk_size, 
                         exe_grok   = exe_grok,
                         exe_hgs    = exe_hgs,
                         )

    def updateParticle(self, part, best, phi1, phi2, iw:int=None, verbose:bool=0): # {{{
        '''
        Explain 

        '''
        limit = self.sources_limits
        nsource = len(part['location'])

        if verbose:
            print(f'updateParticle')
        
        # 위치 업데이트
        # loc  - current location
        # locl - local best location
        # locg - global best location
        # speedg  - local best speed
        old_part    = copy.deepcopy(part)
        old_pb      = copy.deepcopy(part.best)
        old_best    = copy.deepcopy(best)

        nsource = len(part['location'])

        if verbose:
            print(f'updateParticle: check idx = {part.index}')

        for loc_idx in range(nsource):
            pmin, pmax, smin, smax = limit[loc_idx]
            loc   = part['location'][loc_idx]
            locp  = part.best['location'][loc_idx]
            locg  = best['location'][loc_idx]
            
            if verbose:
                print('updateParticle')
                print(f'   loc_idx: {loc_idx}')
                print(f'   loc    : {loc}')
                print(f'   locg   : {locg}')
            # 새로운 위치 샘플링 방식 도입
            mean = [(locp[i] + locg[i]) / 2 for i in range(len(loc))]
            std_dev = [(locp[i] - locg[i]) / 2 for i in range(len(loc))]
            part['location'][loc_idx] = [round(random.gauss(mean[i], std_dev[i])) for i in range(len(loc))]

            if verbose:
                print(f" loc_mean : {mean}")
                print(f" std_dev : {std_dev}")

            # location limit setting
            for i, loc in enumerate(part['location'][loc_idx]):              
                if loc < pmin[i]:
                    part['location'][loc_idx][i] = pmin[i]
                elif loc > pmax[i]:
                    part['location'][loc_idx][i] = pmax[i]

        # flux 업데이트
        for t_idx in range(len(part['flux'])):
            pmin, pmax, smin, smax = limit[t_idx]
            if verbose:
                print(f"t_idx		: {t_idx}")

            flux    = part['flux'][t_idx]
            fluxp   = part.best['flux'][t_idx]
            fluxg   = best['flux'][t_idx]

            if verbose:
                print(f"flux          : {flux}")
            # 새로운 위치 샘플링 방식 도입
            mean_flux = [(fluxp[i] + fluxg[i]) / 2 for i in range(len(flux))]
            std_dev_flux = [(fluxp[i] - fluxg[i]) / 2 for i in range(len(flux))]
            part['flux'][t_idx] = [round(random.gauss(mean_flux[i], std_dev_flux[i]),10) for i in range(len(flux))]

            if verbose:
                print(f" update_flux : {mean_flux}")
                print(f" std_flux : {std_dev_flux}")
            
            for i, flux in enumerate(part['flux'][t_idx]):
                if flux < pmin[3]:
                    part['flux'][t_idx][i] = pmin[3]
                elif flux > pmax[3]:
                    part['flux'][t_idx][i] = pmax[3]
                    
        if verbose:
            print('updateParticle: check update particle.')
            print('location part!')
            print('location : ',old_part['location'], '----','speed : ',old_part.speed_location)
            print('location : ',part['location'], '----','speed : ',part.speed_location)
            print('flux part!')
            print('flux : ',old_part['flux'], '----','speed : ',old_part.speed_flux)
            print('flux : ',part['flux'], '----','speed : ',part.speed_flux)

        part.best = old_pb
        best      = old_best

        return part, part.best, best
        # }}}

