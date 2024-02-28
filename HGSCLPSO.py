#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap 
import random, copy, time, pandas, math, operator, pickle
import tqdm
from .HGSPSO import *
from deap import base, benchmarks, creator, tools

class HgsCLPSO(HgsPSO):
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

    def calculate_pc(self, niter): #{{{
        a = 0.05
        b = 0.45
        ps = self.generation
        i = niter
        pc = a + b * (math.exp(10*(i-1)/(ps-1)) - 1) / (math.exp(10) - 1)
        
        return pc
    #}}}

    def updateParticle(self, part, best, phi1, phi2, Niter,Pc, iw:int=None, verbose:bool=0): # {{{
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
            updated_locl = []

            for i in range(len(loc)):
                if random.random() < Pc:
                    neighbor_best = min([self.pop[idx] for idx in part.neighbors], key=lambda x: x.best.fitness)
                    selected_neighbor = neighbor_best.index
                    updated_locl.append(self.pop[selected_neighbor].best['location'][loc_idx][i])
                    if verbose:
                        print(f"Selected Neighbor Index: {neighbor_best.index}")
                        print(f"Updated Location: {updated_locl[-1]}")
                        print(f"Current Updated Location List: {updated_locl}")
                else:
                    updated_locl.append(part.best['location'][loc_idx][i])
                    if verbose:
                        print("particle best selected")
            locl  = updated_locl
            if verbose:
                print('updateParticle')
                print(f'   loc_idx: {loc_idx}')
                print(f'   loc    : {loc}')
                print(f'   locl   : {locl}')           
                print(f'   locg   : {locg}')

            u1_location = [random.uniform(0, phi1) for _ in loc]
            u2_location = [random.uniform(0, phi2) for _ in loc]
            v_u1_location = [a * (b - c) for a, b, c in zip(u1_location, locl, loc)]
            v_u2_location = [a * (b - c) for a, b, c in zip(u2_location, locg, loc)]
            
            if verbose:
                print('updateParticle: check update speed')
                print(f'   u1_location   {u1_location}')
                print(f'   u2_location   {u2_location}')               
                print(f'   v_u1_location {v_u1_location}')
                print(f'   v_u2_location {v_u2_location}')
                
            if iw is not None: # Iw version
                part.speed_location[loc_idx] = [iw * a + b + c for a, b, c in zip(part.speed_location[loc_idx], v_u1_location, v_u2_location)]
            else:
                part.speed_location[loc_idx] = [a + b + c for a, b, c in zip(part.speed_location[loc_idx], v_u1_location, v_u2_location)]
				
            if verbose:
                print(f' speed : {part.speed_location[loc_idx]} ')

            # 속도 제한 (위치)
            for i, speed in enumerate(part.speed_location[loc_idx]):
                if abs(speed) > smax[i]:
                    part.speed_location[loc_idx][i] = math.copysign(smax[i], speed)
            
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
            updated_fluxl = []
            for i in range(len(flux)):
                if random.random() < Pc:
                    neighbor_best = min([self.pop[idx] for idx in part.neighbors], key=lambda x: x.best.fitness)
                    selected_neighbor = neighbor_best.index
                    updated_fluxl.append(self.pop[selected_neighbor].best['flux'][t_idx][i])
                    if verbose:
                        print(f"Selected Neighbor Index: {neighbor_best.index}")
                        print(f"Updated Location: {updated_fluxl[-1]}")
                        print(f"Current Updated Location List: {updated_fluxl}")
                else:
                    updated_fluxl.append(part.best['flux'][t_idx][i])
                    if verbose:
                        print("particle best selected")
            fluxl  = updated_fluxl

            if verbose:
                print(f"fluxl         : {fluxl}")
                print(f"flux          : {flux}") 
            u1_flux = [random.uniform(0, phi1) for _ in flux]
            u2_flux = [random.uniform(0, phi2) for _ in flux]
            v_u1_flux = [a * (b - c) for a, b, c in zip(u1_flux, fluxl, flux)]
            v_u2_flux = [a * (b - c) for a, b, c in zip(u2_flux, fluxg, flux)]
            
            if verbose:
                print('updateParticle: check update speed')
                print(f'   u1_flux {u1_flux}')
                print(f'   u2_flux {u2_flux}')
                print(f'   v_u1_flux {v_u1_flux}')
                print(f'   v_u2_flux {v_u2_flux}')
            
            if iw:            
                part.speed_flux[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_flux[t_idx], v_u1_flux, v_u2_flux)]
            else:
                part.speed_flux[t_idx] = [a + b + c for a, b, c in zip(part.speed_flux[t_idx], v_u1_flux, v_u2_flux)]
            
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_flux[t_idx]):
                if abs(speed) > smax[3]:
                    part.speed_flux[t_idx][i] = math.copysign(smax[3], speed)
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

    def solve(self,Con_min:int=1.0e-5,criteria:int=50,verbose:bool=None, debug:bool=0,Ncrit:int=0, phi1:int=2, phi2:int=2): # {{{
        '''
        Explain
         solve PSO
        '''
        verbose = self.verbose
        log_path = f'./00_log_result/{self.pso_name}/'
        if verbose:
            gtext = 'SOLVE PART START POINT'
            self.declare(gtext)
            print('solve: Remove all pickle data set!.')
            os.system(f'rm {log_path}*.pkl')
        start_time = time.time()
        toolbox = self.InitToolBox()
        best = None
        Nbest          = math.inf
        i              = 0
        LogPosition = {}
        LogPosition[0] = self.WritePopDict(self.pop, best)
        p1_1,p1_2 = phi1, phi2
        # setting tqdm for update progress
        pbar = tqdm.tqdm(range(self.generation),
                         bar_format='{l_bar}{bar:40}{r_bar}',
                         desc="Processing")
        # Main loop!
        for g in pbar:
            pbar.set_description(f'Processing ({g+1})')
            Niter = g
            best  = self.iteration(Niter,best)
            iw    = self.inertia_weight(Niter)
            p1_1 , p1_2 = self.Adaptive_phi(Niter, p1_1, p1_2)
            self.Neighbor(st='random')
            for idx, p in enumerate(self.pop):
                Pc = self.calculate_pc(Niter)
                # CLPSO version 
                p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,Niter,Pc,iw,verbose=verbose)
            
            # Check the updated best particle information.
            if verbose:
                print('update finished !\n')
                if best is not None:
                    print(f'global best after updated : {best}\n')
                    print(f'global best value         : {best.fitness.values} \n')

            LogPosition[g+1] = self.WritePopDict(self.pop, best)

            # Save the particle data
            if not os.path.exists(f'{log_path}'):
                os.makedirs(f'{log_path}')

            with open(f'{log_path}{self.pso_name}log.pkl','wb') as fid:
                pickle.dump(LogPosition,fid)
            
            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break
  
            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)

            if verbose:
                gtext = 'SOLVE PART END POINT'
                self.declare(gtext)

        # }}}
