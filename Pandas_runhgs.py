#!/usr/bin/env python3
# import specific modules
import numpy as np
import os,sys, glob, shutil
import deap 
import random, copy, time, pandas, math, operator, pickle
import tqdm
import pandas as pd
from deap import base, benchmarks, creator, tools

class HgsPSO:  # {{{
    def __init__(self, prefix, slimits, time_list, obs,
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

        # self parameters of PSO algorithm
        self.slimits        = slimits
        self.npop           = npop
        self.nsource        = len(slimits)
        self.time_list      = time_list
        self.ntime          = len(time_list)
        self.obs            = obs

        self.gpop           = {}
        self.gl_best        = []
        self.ind_best       = []
        
        self.seed           = seed
        
        # setup for PSO name
        self.pso_name       = pso_name
        self.prefix         = prefix
        
        # setup for updateparticle
        self.phi1           = phi1
        self.phi2           = phi2
        
        # other etc...
        self.verbose        = verbose
        self.generation     = generation
        self.nblank         = int(20)

        # set up environment for HGS
        self.exe_grok       = exe_grok
        self.exe_hgs        = exe_hgs
        self.model_dir      = model_dir
        self.tem_path       = tem_path
        
        # set up chunk_size
        self.chunk_size     = chunk_size

        # Set up grid size for location
        self.grid_size      = grid_size

        # initialize population
        self.pop            = {}
        # initialize creator
        self.creator        = creator
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create(
            "Particle",
            dict, 
            fitness=creator.FitnessMin,
            index=None,
            best=None
            )
    # }}}

    def __repr__(self):# {{{
        '''
        show information of HgsPSO
        '''
        string = f'''
npop       : {self.npop}
nsolute    : {self.nsolute}
nsources   : {self.nsource}
        '''
        return string
        # }}}

    def generate_solutes(self,verbose:bool=0, re:bool=0): # {{{
        '''
        Usage 
        solutes = self.generate_solutes()
        '''
        # Save the current random state
        cur_state = random.getstate()
        
        # Declare start point
        if verbose: 
            gtext = 'GENERATE_SOLUTE PART START POINT'
            self.declare(gtext)

        # Set random seed
        random.seed(self.seed)

        if re:
            # save best information
            self.ind_best  = [p.best for p in self.pop]
            # reset dictionary
            self.pop = {} 
        else:
            self.pop = {}

        if verbose:
            print(f'''
Gen solute: check pop reset
{self.pop}''')
            print(f'''
Gen solute: check limit
{self.slimits}''')

        slimits  = self.slimits
        firstkey = next(iter(slimits))
        keys     = slimits[firstkey].keys() 

        for idx in range(self.npop):
            # make instance for each particles
            particles = creator.Particle({})

            for key in keys:
                max_df = slimits['max'][key]
                min_df = slimits['min'][key]
                random_data = {}

                for col in max_df.columns:    
                    max_val = max_df[col].values
                    min_val = min_df[col].values
                    # Make random value between min & max
                    random_data[col] = np.random.uniform(min_val, max_val, size=max_val.shape)

                particles[key] = pd.DataFrame(random_data, index=max_df.index)

            particles.index = idx

            if re:
                particles.best  = self.ind_best[idx]
            else:
                particles.best  = None

            self.pop[idx]   = particles
            
        random.setstate(cur_state)

        if verbose:
            gtext = 'GENERATE_SOLUTE PART END POINT'

        # }}}

    def iteration(self,Niter:int,best,verbose:bool=None): # {{{
        verbose   = self.verbose
        g         = Niter

        if verbose:
            gtext = '   ITERATION PART START POINT'
            self.declare(gtext)
            print(f'iteration : spliting population.')

        spop = self.splitpop(self.chunk_size)

        if verbose:
            print(f'iteartion : split pop in {len(spop)}.')

        # Save each particle information 
        self.gpop[g] = copy.deepcopy(self.pop)

        # Run all particles
        for p in spop:
            self.runHGS(p)

        if verbose:
            print(f"\niteration : {g+1} Times Cost calculation completed\n")

        # Evalulate each cost in individual
        for idx in self.pop: 
            p = self.pop[idx]

            if verbose:
                print(f'iteration : check idx = {idx}')

            # find local best
            if verbose:
                if p.best is not None:
                    print(f'\tlocal best : {p.best}')
                    print(f'\tlocal best : {p.best.fitness.values}\n')

            if verbose:
                if best is not None:
                    print(f'\tglobal best : {best}')
                    print(f'\tglobal best : {best.fitness.values}\n')

            if (p.best is None) or (p.best.fitness < p.fitness):
                #p.best = copy.deepcopy(p)
                if verbose:
                    if g == 0:
                        print('First Generation')
                    if p.best is not None:
                        print(f'\tlocal best before : {p.best}')
                        print(f'\tlocal best before : {p.best.fitness.values}\n')

                p.best = creator.Particle(p)
                p.best.fitness.values = p.fitness.values

                if verbose:
                    if p.best is not None:
                        print(f'\tlocal best after  : {p.best}')
                        print(f'\tlocal best after  : {p.best.fitness.values}\n')

            # find global best.
            if (best is None) or (best.fitness < p.fitness):
                #best = copy.deepcopy(p)
                if verbose:
                    if g == 0:
                        print('First Generation')
                if verbose:
                    if best is not None:
                        print(f'\tglobal best before : {best}')
                        print(f'\tglobal best before : {best.fitness.values}\n')

                best = creator.Particle(p)
                best.fitness.values = p.fitness.values

                if verbose:
                    if best is not None:
                        print(f'\tglobal best after  : {best}')
                        print(f'\tglobal best after  : {best.fitness.values}\n')       
        if verbose:
            print("iteration : Update Chunk_size local populations\n")
            gtext = '   ITERATION PART END POINT'
            self.declare(gtext)
        
        return best # }}}

    def Neighbor(self,st:str='pass'): # {{{
        '''
        Define neighbor strategy
        1. ring
        2. niching
        3. pass
        '''
        n = self.npop
        verbose = self.verbose

        if st == 'ring':
            # Set neighbors in a ring topology using index
            n = self.npop
            for particle in self.pop:
                idx = particle.index
                num_neighbors = int(0.1 * n)
                start_idx = idx - num_neighbors // 2
                neighbors_idx = [(start_idx + i) % n for i in range(num_neighbors)]
                particle.neighbors = neighbors_idx  # Save neighbors' index

                if verbose:
                    print(f"generate : {particle} ")
                    print(f"generate : {particle.neighbors} ")

        elif st == 'niching':
            # Set neighbors in a niching method using index
            pass

        elif st == 'random':
            for p in self.pop:
                idx = p.index
                p_neighbors = [i for i in range(self.npop) if i != idx]
                # Randomly select two neighbors
                s_neighbors = random.sample(p_neighbors, 2)
                # Save selected neighbors' index
                p.neighbors = s_neighbors
                if verbose:
                    print(f"selected Neighbor : {s_neighbors} ")
        else:
            pass  # }}}
  
    def splitpop(self,chunk_size:int=None):# {{{
        '''
        split list!
        '''
        if chunk_size == None:
            chunk_size = self.chunk_size

        items = list(self.pop.items())

        new_list = []

        for i in range(0,len(items), chunk_size):
            new_list.append(dict(items[i:i+chunk_size]))

        return new_list
        # }}}

    def updateParticle(self, part, best, phi1, phi2, iw:int=None, verbose:bool=0): # {{{
        '''
        Explain 
        Update each particle's positions.
        '''
        if verbose:
            gtext = 'UPDATE_PARTICLE PART START POINT'
            self.declare(gtext)
        
        '''
        loc  - current location
        locl - local best location
        locg - global best location
        speedg  - local best speed
        '''
        old_part    = copy.deepcopy(part)
        old_pb      = copy.deepcopy(part.best)
        old_best    = copy.deepcopy(best)

        random.seed(self.seed)

        if verbose:
            print(f'updateParticle : check idx = {part.index}')
            print(f' PART        : {part}')
            print(f' PART BEST   : {part.best}')
            print(f' GLOBAL BEST : {best}')
            print(f' Fitness local : {part.fitness.values}')
            print(f' Fitness Global : {best.fitness.values}')

        limits  = self.slimits

        keys     = part.keys()
        par = []
        for key in part.keys():
            if 'speed' in key:
                pass
            else:
                par.append(key)

        for i in par:
            if verbose:
                print(f'Parameter: {i}')

            for j in part[i].index:
                if verbose:
                    print(f'Nsource: {j}')

                target = part[i].loc[j]
                Gbest  = best[i].loc[j]
                Lbest  = part.best[i].loc[j]
                speed  = part[f'{i}_speed'].loc[j]

                max_val   = limits['max'][i].loc[j]
                max_speed = limits['max'][f'{i}_speed'].loc[j]
                min_val   = limits['min'][i].loc[j]
                min_speed = limits['min'][f'{i}_speed'].loc[j]

                u1 = np.random.uniform(0, phi1, len(target))
                u2 = np.random.uniform(0, phi2, len(target))

                v_u1 = [a * (b - c) for a, b, c in zip(u1, Lbest, target)]
                v_u2 = [a * (b - c) for a, b, c in zip(u2, Gbest, target)]

                if iw is not None:
                    sp = [iw * a + b + c for a, b, c in zip(speed, v_u1, v_u2)]
                else:
                    sp = [a + b + c for a, b, c in zip(speed, v_u1, v_u2)]

                for sidx, ms in enumerate(max_speed):
                    if abs(sp[sidx]) > abs(ms):
                        sp[sidx] = math.copysign(ms, sp[sidx])

                part[f'{i}_speed'].loc[j] = sp

                if verbose:
                    print(f'speed : {sp}')
                    print(f'target: {target}')

                target = target + sp

                for (tidx, maxs), mins in zip(enumerate(max_val), min_val):
                    if target[tidx] > maxs:
                        target[tidx] = maxs
                    elif target[tidx] < mins:
                        target[tidx] = mins

                if verbose:
                    print(target)

                part[i].loc[j] = target

        part.best = old_pb
        best      = old_best

        return part, part.best, best
        # }}}

    def InitToolBox(self): # {{{
        # Toolbox
        toolbox = base.Toolbox()
        toolbox.register("particle", tools.initIterate, self.generate_solutes)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.particle)
        toolbox.register("update_particle", self.updateParticle)
        toolbox.register("evaluate", self.runHGS)

        return toolbox
        # }}}

    def read_tem(self, replacements): # {{{
        with open(self.tem_path, 'r') as file:
            template_cont = file.read()

        for key, value in replacements.items():
            template_cont = template_cont.replace(f'# {key} #', str(value))

        return template_cont
# }}}

    def runHGS(self,population): # {{{
        
        # initialize variables # {{{
        prefix     = self.prefix # debugging
        verbose    = self.verbose
        local_npop = len(population)
        pso_name   = self.pso_name
        obs        = self.obs
        timelist   = self.time_list
        ntime      = len(timelist)
        model_dir  = self.model_dir

        # setup initial grok and hgs name
        exe_grok = self.exe_grok
        exe_hgs = self.exe_hgs
        dest_dir = f'./particles/{pso_name}'  # particle_index 추가

        if verbose:
            gtext = '       RUNHGS PART START POINT'
            self.declare(gtext)

        # make destination directory.
        if not os.path.exists(dest_dir):
            print(f'make directory {dest_dir}')
            shutil.copytree(f'{model_dir}', dest_dir,dirs_exist_ok=True)
        else:
            command = f'cd {dest_dir} && ./clearall.sh'
            os.system(command)
            pass # }}}

        if verbose: print('runHGS: Write contaminant_list information')  # {{{
        CNum_string     = ''
        Cinclude_string = ''
        C_lines         = ''
        
        # New string for initial concentration
        Cinit_string = "choose nodes all\ninitial concentration\n"
        for p in population:
            pindex = p
            Cfname = f'./01_inc/Contaminant_{pindex}.inc'
            lines = ''
            CNum_string += f'''
solute
    name
    PCE{pindex}
end\n'''

            Cinclude_string += f'\ninclude {Cfname}\n'
            
        for i in range(local_npop):
            Cinit_string += "1.0e-7\n"
        Cinit_string += "clear chosen nodes\n"

        fname = os.path.join(dest_dir, '01_inc/Contaminant_list.inc')
        with open(fname, 'w') as fid:
            fid.write(CNum_string)
            fid.write(Cinit_string)
            fid.write(Cinclude_string)
        # }}}
        
        if verbose: print('runHGS: Write contamination') # {{{

        for idx, p in population.items():
            fluxarray = np.zeros(local_npop)
            firstkey  = next(iter(p))
            pindex    = idx # global index in array
            Cfname = f'./01_inc/Contaminant_{pindex}.inc'
            # for time step iteration
            C_lines = ''
            C_tem   = ''
            
            for sidx in p[firstkey].index:
                replacements = {'L1': p['loc'].loc[sidx]['x'],
                                'L2': p['loc'].loc[sidx]['y'],
                                'L3': p['loc'].loc[sidx]['z'],
                                'ntime': ntime}
                C_tem = '\n'+self.read_tem(replacements)+'\n'

                # Write sources with the same flux value for each timestep
                for flux, time in zip(p['flux'].loc[sidx], timelist):
                    fidx = pindex%local_npop
                    C_tem += f'{time[0]} {time[1]}' 
                    fluxarray[fidx]=flux
                    fluxstr = ' '.join(map(str, fluxarray))
                    C_tem += f' {fluxstr}'
                    C_tem += '\n'
                C_lines += C_tem
            # set contamination name
            with open(os.path.join(dest_dir,Cfname),'w') as fid:
                fid.write(C_lines)
        # }}}
            
        retry_count = 0
        max_retries = 5

        while retry_count <= max_retries: # {{{
            if verbose: 
                print('runHGS: Run HGS.') 

            command = f'cd {dest_dir} && {exe_grok} > null && {exe_hgs} > null'
            exit_code = os.system(command)

            if exit_code == 0:
                if verbose:
                    print('runHGS: HGS excuted successfully')
                break
            else:              
                if verbose: 
                    print('runHGS: An error occurred. Retrying...')
            retry_count += 1

        if retry_count > max_retries:
            print('runHGS: Max retries reached. Stopping...')
            sys.exit("Error: Maximum number of retries reached. Exiting the code.")

            # }}}
        if verbose: 
            print(f'runHGS: Get results')  # {{{

        total_cost = np.zeros((local_npop,))
        for idx, p in population.items():
            pindex = idx%local_npop
            for well_name in obs.keys():

                obs_time = obs[well_name]['time']
                obs_conc = obs[well_name]['conc']

                # 해당 입자, 용액, 및 well_name에 대한 파일 패턴 찾기
                fname = os.path.join(dest_dir, f'{prefix}o.observation_well_conc.{well_name}.PCE{idx}.dat')
                if not os.path.isfile(fname):
                    raise Exception(f'ERROR: we cannot find {fname}')
                    
                model_ts   = np.loadtxt(fname, skiprows=25)
                model_time = model_ts[:,0]
                model_conc = model_ts[:,1]

                for otime, oconc in zip(obs_time, obs_conc):
                    pos  = (otime == model_time)
                    mconc = model_conc[pos] # find specific time concentration.
                    total_cost[pindex] += (mconc-oconc)**2

            if verbose:
                print(f'runHGS: Total cost \n {total_cost}') # }}}

        if verbose: 
            print('runHGS: assign all cost in each poplulation')

        for (idx ,p), cost in zip(population.items(), total_cost):
            self.pop[idx].fitness.values = (cost,)

        if verbose:
            gtext = '       RUNHGS PART END POINT'
            self.declare(gtext) # }}}

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

        # Variable setting
        toolbox        = self.InitToolBox()
        best           = None
        Nbest          = math.inf
        i              = 0
        LogPosition    = {}
        LogPosition[0] = self.WritePopDict(self.pop, best)
        p1_1,p1_2      = phi1, phi2
        
        # setting tqdm for update progress
        pbar = tqdm.tqdm(range(self.generation),
                         bar_format='{l_bar}{bar:40}{r_bar}',
                         desc="Processing")
        # Main loop!
        for g in pbar:
            pbar.set_description(f'Processing ({g+1})')
            Niter         = g
            self.ind_best = []
            self.gl_best  = []

            if debug:
                self.verbose=0
            if verbose:
                print(f' SOLVE : Generation == No.{g} ==')

            # update BEST and runHGS
            best  = self.iteration(Niter,best)
            # !!==
            for idx, p in self.pop.items():
                if verbose:
                    print(f' UPDATE PART: {idx}')
                if verbose:
                    print(f' Cur particle:{p}')
                # original version
                p, p.best, best = self.updateParticle(p, best,p1_1,p1_2, verbose=self.verbose)
                if verbose:
                    print(f'Updated particle: {p}')
                    print(f'check self.pop: {self.pop}')
            
            if debug:
                self.verbose=1
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

    def WritePopDict(self, pop,best): # {{{
        gbest  = None
        gvalue = None
        pop    = None
        
        if best is not None:
            gbest  = dict(best)
            gvalue = best.fitness.values[0]
            pop    = self.pop
        
            result = dict(pop=copy.deepcopy(pop),
                        gbest  = gbest,
                        gvalue = gvalue,
                        )
        else:
            result = None

        return result
    # }}}
    
    def iteration(self,Niter:int,best,verbose:bool=None): # {{{
        verbose   = self.verbose
        g         = Niter

        if verbose:
            gtext = '   ITERATION PART START POINT'
            self.declare(gtext)
            print(f'iteration : spliting population.')

        spop = self.splitpop(self.chunk_size)

        if verbose:
            print(f'iteartion : split pop in {len(spop)}.')

        # Save each particle information 
        self.gpop[g] = copy.deepcopy(self.pop)

        # Run all particles
        for p in spop:
            self.runHGS(p)

        if verbose:
            print(f"\niteration : {g+1} Times Cost calculation completed\n")

        # Evalulate each cost in individual
        for idx in self.pop: 
            p = self.pop[idx]

            if verbose:
                print(f'iteration : check idx = {idx}')

            # find local best
            if verbose:
                if p.best is not None:
                    print(f'\tlocal best : {p.best}')
                    print(f'\tlocal best : {p.best.fitness.values}\n')

            if verbose:
                if best is not None:
                    print(f'\tglobal best : {best}')
                    print(f'\tglobal best : {best.fitness.values}\n')

            if (p.best is None) or (p.best.fitness < p.fitness):
                #p.best = copy.deepcopy(p)
                if verbose:
                    if g == 0:
                        print('First Generation')
                    if p.best is not None:
                        print(f'\tlocal best before : {p.best}')
                        print(f'\tlocal best before : {p.best.fitness.values}\n')

                p.best = creator.Particle(p)
                p.best.fitness.values = p.fitness.values

                if verbose:
                    if p.best is not None:
                        print(f'\tlocal best after  : {p.best}')
                        print(f'\tlocal best after  : {p.best.fitness.values}\n')

            # find global best.
            if (best is None) or (best.fitness < p.fitness):
                #best = copy.deepcopy(p)
                if verbose:
                    if g == 0:
                        print('First Generation')
                if verbose:
                    if best is not None:
                        print(f'\tglobal best before : {best}')
                        print(f'\tglobal best before : {best.fitness.values}\n')

                best = creator.Particle(p)
                best.fitness.values = p.fitness.values

                if verbose:
                    if best is not None:
                        print(f'\tglobal best after  : {best}')
                        print(f'\tglobal best after  : {best.fitness.values}\n')       
        if verbose:
            print("iteration : Update Chunk_size local populations\n")
            gtext = '   ITERATION PART END POINT'
            self.declare(gtext)
        
        return best # }}}
    
    def inertia_weight(self,Niter): # {{{

        iw = 0.9 - Niter * (0.7/self.generation)

        return iw # }}}

    def Adaptive_phi(self,Niter,phi1,phi2): # {{{

        phi1 = phi1 - Niter * (2/self.generation)
        phi2 = phi2 + Niter * (2/self.generation)

        return phi1, phi2 # }}}

    def regen(self,Nbest,best,i,Ncrit): # {{{
        if Nbest > best.fitness.values[0]:
            Nbest = best.fitness.values[0]
            i = 0
        else:
            i += 1
        if i == Ncrit:
            self.seed = None
            self.generate_solutes(re=1)
            if self.verbose:
                gtext = 'REGENERATE PARTICLES'
                self.declare(gtext)
        return i,Nbest
    # }}}

    def declare(self, gtext): # {{{
        print(f'''
{'=' * self.nblank * 4} 
{' ' * self.nblank} --- {gtext} --- 
{'=' * self.nblank * 4}
''') # }}}

