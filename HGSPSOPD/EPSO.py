#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap
import random, copy, time, pandas, math, operator, pickle
import tqdm
import pandas as pd
from scipy.optimize import linear_sum_assignment
from deap import base, benchmarks, creator, tools
from .HGSPSO    import *
from .HGSCLPSO  import *
from .HGSFDR    import *
from .HGSLIPs   import *
from .HGSTVAC   import *

class EPSO(HgsPSO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self,Con_min:int=1.0e-5,criteria:int=50,verbose:bool=None, debug:bool=0): # {{{
        '''
        Explain
         solve PSO
        '''
        verbose = self.verbose
        log_path = f'./00_log_result/{self.pso_name}/'

        # Variable setting
        toolbox        = self.InitToolBox()
        best           = None
        Nbest          = math.inf
        i              = 0
        LogPosition    = {}
        Logbest        = {}
        Logfitvalue    = {}

        # setting tqdm for update progress
        pbar = tqdm.tqdm(range(self.generation),
                         bar_format='{l_bar}{bar:40}{r_bar}',
                         desc="Processing")
        if verbose:
            gtext = 'SOLVE PART START POINT'
            self.declare(gtext)
            print('solve: Remove all pickle data set!.')
            os.system(f'rm {log_path}*.pkl')

        N_st  = 5 
        fit_c = 0
        LP    = 50
        s_mem = np.zeros(N_st)
        f_mem = np.zeros(N_st)
        pk    = np.ones(N_st) / N_st

        column_names = ['Generation'] + [f'pk_{i+1}' for i in range(N_st)] + [f'sk_{i+1}' for i in range(N_st)]
        column_names += [f's_mem_{i+1}' for i in range(N_st)] + [f'f_mem_{i+1}' for i in range(N_st)]
        ps_df = pandas.DataFrame(columns=column_names)
        current_fitness = None
        st_array  = np.random.randint(0, 5, size=self.npop, dtype=int)
        fit_array = np.zeros(self.npop, dtype=float)

        # Main loop!
        for g in pbar:
            pbar.set_description(f'Processing ({g+1})')
            self.g         = g

            if debug:
                self.verbose=0
            if verbose:
                print(f' SOLVE : Generation == No.{g} ==')

            if fit_c % LP == 0 and fit_c > 0: 
                total = s_mem + f_mem
                total[total == 0] = 1  # Avoid division by zero
                sk = (s_mem / total) + 0.01
                pk = sk / np.sum(sk)
                # Create a new row and fill it
                new_row = {'Generation': g}
                for i in range(N_st):
                    new_row[f'pk_{i+1}'] = pk[i]
                    new_row[f'sk_{i+1}'] = sk[i]
                    new_row[f's_mem_{i+1}'] = s_mem[i]
                    new_row[f'f_mem_{i+1}'] = f_mem[i]

                # Add the new row to the DataFrame
                N_row = pandas.DataFrame([new_row])
                ps_df = pandas.concat([ps_df, N_row], ignore_index=True)

                if verbose:
                    print(f's_mem: {s_mem}')
                    print(f'f_mem: {f_mem}')
                    print(f'sk   : {sk}')
                    print(f'pk   : {pk}')
                # Reset memory
                s_mem = np.zeros(N_st)
                f_mem = np.zeros(N_st)

            # update BEST and runHGS
            best = self.iteration(self.g,best)
            self.best = best
            # Change index
            if g == 0:
                pass
            else:
                if verbose:
                    gtext = 'INDEXING PART POINT'
                    self.declare(gtext)
                self.indexing()
            # adaptive & iw calculation
            if verbose:
                gtext = 'ESE PART POINT'
                self.declare(gtext)
            if g == 0:
                f = np.inf
                pass
            else:
                f = self.EvolutionaryStateEstimation()

            self.logc1[g] = self.c1
            self.logc2[g] = self.c2
            self.logiw[g] = self.w
            self.logf[g]  = f

            if verbose:
                print(f' C1, C2, iw : {self.c1} / {self.c2} / {self.w}')

            #=======
            for idx, p in self.pop.items():
                if verbose:
                    print(f'st_array : \n{st_array}')
                    print(f'fit_array : {fit_array}')
                current_fitness = fit_array[idx]
                # Check if the fitness value has improved (decreased in this case)
                if current_fitness is not None:  # Skip first generationa
                    st_selected = int(st_array[idx])
                    if verbose:
                        print(f' idx        : {idx}')
                        print(f'st_selected : {st_selected}')
                        print(f'Before fitness value  : {current_fitness}')
                        print(f'Current fitness_value : {p.fitness.values[0]}')
                    if p.fitness.values[0] < current_fitness:
                        s_mem[st_selected] += 1
                        if verbose:
                            print(f'Strategy       : {st_selected}')
                            print(f'Success memory : {s_mem[st_selected]}')
                    else:
                        f_mem[st_selected] += 1
                        if verbose:
                            print(f'Strategy       : {st_selected}')
                            print(f'failure memory : {f_mem[st_selected]}')

            if verbose:
                print(f' C1, C2, iw : {self.c1} / {self.c2} / {self.w}')

            for idx, part in self.pop.items():
                if verbose:
                    gtext = 'UPDATE PART POINT'
                    self.declare(gtext)
                # Select a strategy based on probability
                st_selected   = np.random.choice(range(N_st), p=pk)
                if verbose:
                    print(f'st_selected : {st_selected}')
                st_array[idx] = st_selected
                if verbose:
                    print(f'st_array after! : \n{st_array}')
                # Store the current fitness value before updating the particle
                fit_array[idx]   = part.fitness.values[0]
                if verbose:
                    print(f'fit_list after! : \n{fit_array}')
                if debug:
                    verbose = 0
                if st_selected == 0:  # Normal version
                    if self.verbose:
                        gtext = ' PSO selected'
                        self.declare(gtext)
                    HgsPSO.updateParticle(self,part)
                elif st_selected == 1:  # iw version
                    if self.verbose:
                        gtext = ' FDR selected'
                        self.declare(gtext)
                    HgsFDR.updateParticle(self,part)
                elif st_selected == 2: # Iw,Adaptive version
                    if self.verbose:
                        gtext = ' TVAC selected'
                        self.declare(gtext)
                    HgsTVAC.updateParticle(self,part)
                elif st_selected == 3: # CFPSO    
                    if self.verbose:
                        gtext = ' LIPs selected'
                        self.declare(gtext)
                    HgsLIPs.updateParticle(self,part)
                else: # PSO
                    if self.verbose:
                        gtext = ' CLPSO selected'
                        self.declare(gtext)
                    HgsCLPSO.updateParticle(self,part)
            # Check the updated best particle information.
            if verbose:
                print('update finished !\n')
                if best is not None:
                    print(f'global best after updated : {best}\n')
                    print(f'global best value         : {best.fitness.values} \n')

            LogPosition[g+1]  = self.WritePopDict(best, self.pop)
            Logbest[g+1]      = self.WritePopDict(best, best)
            bestfitval        = {'gbfit': best.fitness.values[0]}
            Logfitvalue[g+1]  = self.WritePopDict(best, bestfitval)
            
            # Save the particle data
            if not os.path.exists(f'{log_path}'):
                os.makedirs(f'{log_path}')

            # Copy save insert part
            self.backup(log_path)

            with open(f'{log_path}{self.pso_name}log.pkl','wb') as fid:
                pickle.dump(LogPosition,fid)

            with open(f'{log_path}{self.pso_name}best.pkl','wb') as fid:
                pickle.dump(Logbest,fid)

            with open(f'{log_path}{self.pso_name}fit.pkl','wb') as fid:
                 pickle.dump(Logfitvalue,fid)

            with open(f'{log_path}{self.pso_name}c1log.pkl','wb') as fid:
                pickle.dump(self.logc1,fid)

            with open(f'{log_path}{self.pso_name}c2log.pkl','wb') as fid:
                pickle.dump(self.logc2,fid)            

            with open(f'{log_path}{self.pso_name}iwlog.pkl','wb') as fid:
                pickle.dump(self.logiw,fid)

            with open(f'{log_path}{self.pso_name}flog.pkl','wb') as fid:
                pickle.dump(self.logf,fid)

            with open(f'{log_path}{self.pso_name}initial.pkl','wb') as fid:
                pickle.dump(self.initial,fid)

            ps_df.to_csv(f'{log_path}Pk_Sk_values.csv',index=False)           

            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break

            if self.Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, self.Ncrit)
 
        # }}}
