#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap 
import random, copy
import time
import pandas
import math
import operator
import pickle
from deap import base, benchmarks, creator, tools
HOMEDIR = os.getenv('HOME')
sys.path.append(os.path.join(HOMEDIR, 'data/hgs/01_HGSPSO_module/'))
import hgspso
from hgspso import PSO_k

class EPSO(PSO_k.HgsPSO):
    def solve(self,LP:int=50,Con_min:int=1.0e-5,criteria:int=50,verbose:bool=None, debug:bool=0,Ncrit:int=0, phi1:int=2, phi2:int=2): # {{{
        '''
        Explain
         solve PSO
        '''
        verbose = self.verbose
        log_path = f'./00_log_result/{self.pso_name}/'
        if verbose:
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
        p2_1,p2_2 = 2.1,2
        
        N_st  = 6 
        fit_c = 0
        s_mem = np.zeros(N_st)
        f_mem = np.zeros(N_st)
        pk    = np.ones(N_st) / N_st
        column_names = ['Generation'] + [f'pk_{i+1}' for i in range(N_st)] + [f'sk_{i+1}' for i in range(N_st)]
        column_names += [f's_mem_{i+1}' for i in range(N_st)] + [f'f_mem_{i+1}' for i in range(N_st)]
        ps_df = pandas.DataFrame(columns=column_names)
        current_fitness = None
        st_array  = np.random.randint(0, 6, size=self.npop, dtype=int)
        fit_array = np.zeros(self.npop, dtype=float)
        # this is main loop!
        for g in range(self.generation):
            # Update strategy probabilities every LP iterations
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

            Niter = g
            best  = self.iteration(Niter,best)
            iw    = self.inertia_weight(Niter)
            p1_1 , p1_2 = self.Adaptive_phi(Niter, p1_1, p1_2)
            fit_c += 1
            self.Neighbor(st='random')
            for idx, p in enumerate(self.pop):
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
                # Select a strategy based on probability
                st_selected   = np.random.choice(range(N_st), p=pk)
                if verbose:
                    print(f'st_selected : {st_selected}')
                st_array[idx] = st_selected
                if verbose:
                    print(f'st_array after! : \n{st_array}')
                # Store the current fitness value before updating the particle
                fit_array[idx]   = p.fitness.values[0]
                if verbose:
                    print(f'fit_list after! : \n{fit_array}')
                if debug:
                    verbose = 0
                if st_selected == 0:  # Normal version
                    p, p.best, best = self.updateParticle(p, best, phi1, phi2, verbose=verbose)
                elif st_selected == 1:  # iw version
                    p, p.best, best = self.updateParticle(p, best, p1_1, p1_2, verbose=verbose)
                elif st_selected == 2: # Iw,Adaptive version
                    p, p.best, best = self.updateParticle(p, best, p1_1, p1_2, iw, verbose=verbose)
                elif st_selected == 3: # CFPSO    
                    p, p.best, best = PSO_k.HgsCFPSO.updateParticle(self,p, best, p2_1, p2_2, iw, verbose=verbose)
                elif st_selected == 4: # BBPSO
                    p, p.best, best = PSO_k.HgsBBPSO.updateParticle(self,p, best, p1_1, p1_2, iw, verbose=verbose)
                else: # CLPSO
                    Pc = PSO_k.HgsCLPSO.calculate_pc(self,Niter)
                    p, p.best, best = PSO_k.HgsCLPSO.updateParticle(self,p, best, p1_1, p1_2,Niter,Pc, iw, verbose=verbose)

                if debug:
                    verbose = 1
                    
            
            # Check the updated best particle information.
            if verbose:
                print('update finished !\n')
                if best is not None:
                    print(f'global best after updated : {best}\n')
                    print(f'global best value         : {best.fitness.values} \n')

            LogPosition[g+1] = self.WritePopDict(self.pop, best)

            # Save the algorithm end time
            end_time = time.time()

            # Calculate algorithm consume time
            execution_time = end_time - start_time

            print("\n-----Execution time:----- ", execution_time, "seconds")
            
            # Save the particle data
            if not os.path.exists(f'{log_path}'):
                os.makedirs(f'{log_path}')
            with open(f'{log_path}{self.pso_name}log.pkl','wb') as fid:
                pickle.dump(LogPosition,fid)
            ps_df.to_csv(f'{log_path}Pk_Sk_values.csv',index=False)           
            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break

            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)
 
        return LogPosition, execution_time
        # }}}
