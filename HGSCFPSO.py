#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap 
import random, copy, time, pandas, math, operator, pickle
import tqdm
from .HGSPSO import *
from deap import base, benchmarks, creator, tools

class HgsCFPSO(HgsPSO):
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
        loc  - current location
        locl - local best location
        locg - global best location
        speedg  - local best speed
        '''
        phi = phi1+phi2
        K = 2 / abs(2 - phi - math.sqrt(phi ** 2 - 4 * phi))
        if verbose:
            print(f"update_particle : K value {K}")

        limit = self.sources_limits
        nsource = len(part['location'])

        if verbose:
            print(f'updateParticle')
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

            u1_location = [random.uniform(0, phi1) for _ in loc]
            u2_location = [random.uniform(0, phi2) for _ in loc]
            v_u1_location = [a * (b - c) for a, b, c in zip(u1_location, locp, loc)]
            v_u2_location = [a * (b - c) for a, b, c in zip(u2_location, locg, loc)]
            
            if verbose:
                print('updateParticle: check update speed')
                print(f'   v_u1_location {v_u1_location}')
                print(f'   v_u2_location {v_u2_location}')
                print(f'   u1_location   {u1_location}')
                print(f'   u2_location   {u2_location}')
                
            if iw is not None: # Iw version
                part.speed_location[loc_idx] = [iw * a + b + c for a, b, c in zip(part.speed_location[loc_idx], v_u1_location, v_u2_location)]
            else:
                part.speed_location[loc_idx] = [a + b + c for a, b, c in zip(part.speed_location[loc_idx], v_u1_location, v_u2_location)]
				
            if verbose:
                print(f' speed K before: {part.speed_location}')
            
            # Constriction Factor Calculation
            CF_speed_loc = [K*speed for speed in part.speed_location[loc_idx]]
            part.speed_location[loc_idx] = CF_speed_loc
            if verbose:
                print(f' speed K after : {part.speed_location}')
            
            # 속도 제한 (위치)
            for i, speed in enumerate(part.speed_location[loc_idx]):
                if abs(speed) > smax[i]:
                    part.speed_location[loc_idx][i] = math.copysign(smax[i], speed)
            if verbose:
            	for speed_loc in part.speed_location[loc_idx]:
                    print(f'speed_loc after CF calculation : \n{speed_loc}')
            
            # Update local position, speed
            part['location'][loc_idx] = [round(a + b) for a, b in zip(part['location'][loc_idx], part.speed_location[loc_idx])]

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
            u1_flux = [random.uniform(0, phi1) for _ in flux]
            u2_flux = [random.uniform(0, phi2) for _ in flux]
            v_u1_flux = [a * (b - c) for a, b, c in zip(u1_flux, fluxp, flux)]
            v_u2_flux = [a * (b - c) for a, b, c in zip(u2_flux, fluxg, flux)]
            
            if verbose:
                print('updateParticle: check update speed')
                print(f'   u1_flux {u1_flux}')
                print(f'   u2_flux {u2_flux}')
                print(f'   v_u1_location {v_u1_flux}')
                print(f'   v_u2_location {v_u2_flux}')
            
            if iw:            
                part.speed_flux[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_flux[t_idx], v_u1_flux, v_u2_flux)]
            else:
                part.speed_flux[t_idx] = [a + b + c for a, b, c in zip(part.speed_flux[t_idx], v_u1_flux, v_u2_flux)]
            
            if verbose:
                print(f'speed_flux before calculation : \n{part.speed_flux[t_idx]}')
            # Constriction Factor Calculation
            CF_speed_flux = [K*speed for speed in part.speed_flux[t_idx]]
            part.speed_flux[t_idx] = CF_speed_flux
            if verbose:
            	print(f'speed_flux after calculation : \n{part.speed_flux[t_idx]}')
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_flux[t_idx]):
                if abs(speed) > smax[3]:
                    part.speed_flux[t_idx][i] = math.copysign(smin[3], speed)
            if verbose:
                print(f'    speed {part.speed_flux[t_idx]}')
            
            # Update local flux , speed
            part['flux'][t_idx] = [round(a + b, 10) for a, b in zip(part['flux'][t_idx], part.speed_flux[t_idx])]
            
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

