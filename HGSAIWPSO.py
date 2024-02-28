#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap 
import random, copy, time, pandas, math, operator, pickle
import tqdm
from .HGSPSO import *
from deap import base, benchmarks, creator, tools

class HgsAIWPSO(HgsPSO):

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

    def solve(self,Con_min:int=1.0e-5,criteria:int=50,verbose:bool=None, debug:bool=0,Ncrit:int=0, phi1:int=2, phi2:int=2): 
# {{{
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
            for idx, p in enumerate(self.pop):
                # None iw version
                #p, p.best, best = self.updateParticle(p, best,p1_1,p1_2, verbose=verbose)
                # iw version
                #p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,iw,verbose=verbose)
                # Adaptive version
                p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,iw,verbose=verbose)
            
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

