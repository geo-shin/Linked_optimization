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

class HgsPSO:  # {{{
    def __init__(self, prefix, slimits, time_list, obs,
            seed:int       =45,
            model_dir:str  ='base_model',
            tem_path:str   ='tem_path',
            generation:int =1,
            npop:int       =100,
            verbose:bool   =True, 
            pso_name:str   ='pso',
            c1:int         = 1.5,
            c2:int         = 1.5,
            w_max:int      = 0.9,
            w_min:int      = 0.4,
            grid_size:int  = 100,
            chunk_size:int = 10,
            ):

        # self parameters of PSO algorithm
        self.slimits        = slimits
        self.npop           = npop
        self.time_list      = time_list
        self.ntime          = len(time_list)
        self.obs            = obs
        self.gpop           = {}
        self.seed           = seed
        self.best           = None
        self.FDRbest        = None
        # setup for PSO name
        self.pso_name       = pso_name
        self.prefix         = prefix
        
        # other etc...
        self.verbose        = verbose
        self.generation     = generation
        self.nblank         = int(20)

        # set up environment for HGS
        self.model_dir      = model_dir
        self.tem_path       = tem_path

        # set up control parameters
        self.c1             = c1
        self.c2             = c2
        self.w_min          = w_min
        self.w_max          = w_max
        self.w              = w_max
        self.Previous_State = 'S1'
        self.rule_base      = pd.DataFrame(data=[['S3', 'S2', 'S2', 'S1', 'S1', 'S1', 'S4'],
                                            ['S3', 'S2', 'S2', 'S2', 'S1', 'S1', 'S4'],
                                            ['S3', 'S3', 'S2', 'S2', 'S1', 'S4', 'S4'],
                                            ['S3', 'S3', 'S2', 'S1', 'S1', 'S4', 'S4']])
        self.rule_base.columns = ['S3', 'S3&S2', 'S2', 'S2&S1', 'S1', 'S1&S4', 'S4']
        self.rule_base.index   = ['S1', 'S2', 'S3', 'S4']
        self.logf  = np.zeros(self.generation)
        self.logc1 = np.zeros(self.generation)
        self.logc2 = np.zeros(self.generation) 
        self.logiw = np.zeros(self.generation) 

        # set up chunk_size
        self.chunk_size     = chunk_size

        # Set up grid size for location
        self.grid_size      = grid_size

        # initialize population
        self.pop            = {}
        self.par            = []
        self.source         = []
        self.columns        = []
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
        '''
        return string # }}}

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

        for par in keys:
            if 'speed' in par:
                pass
            else:
                self.par.append(par)
        self.source = slimits[firstkey][par].index

        for idx in range(self.npop):
            # make instance for each particles
            particles = creator.Particle({})

            for key in keys:
                max_df = slimits['max'][key]
                min_df = slimits['min'][key]
                random_data = {}

                for col in max_df.columns:    
                    self.columns.append(col)
                    columns      = set(self.columns)
                    self.columns = list(columns)
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
        
        return best # }}}

    def updateParticle(self): # {{{
        '''
        Update each particle's positions.
        '''
        random.seed(self.seed)

        c1 = self.c1
        c2 = self.c2

        best    = self.best
        limits  = self.slimits
        par     = self.par

        for idx, part in self.pop.items():
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
                    speed  = part[f'{i}_speed'].loc[j]

                    max_val   = limits['max'][i].loc[j]
                    max_speed = limits['max'][f'{i}_speed'].loc[j]
                    min_val   = limits['min'][i].loc[j]
                    min_speed = limits['min'][f'{i}_speed'].loc[j]

                    u1 = np.random.uniform(0, c1, len(target))
                    u2 = np.random.uniform(0, c2, len(target))

                    v_u1 = [a * (b - c) for a, b, c in zip(u1, Lbest, target)]
                    v_u2 = [a * (b - c) for a, b, c in zip(u2, Gbest, target)]

                    sp = [(self.w * a) + b + c for a, b, c in zip(speed, v_u1, v_u2)]

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

        if self.verbose:
            print(f' pop updated: {self.pop}') # }}}

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

            command = f'cd {dest_dir} && grok > null && hgs > null'
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

    def solve(self,Con_min:int=1.0e-5,criteria:int=50,verbose:bool=None, debug:bool=0,Ncrit:int=0): # {{{
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
        LogPosition[0] = self.WritePopDict(self.pop, best)
        
        # setting tqdm for update progress
        pbar = tqdm.tqdm(range(self.generation),
                         bar_format='{l_bar}{bar:40}{r_bar}',
                         desc="Processing")
        if verbose:
            gtext = 'SOLVE PART START POINT'
            self.declare(gtext)
            print('solve: Remove all pickle data set!.')
            os.system(f'rm {log_path}*.pkl')

        # Main loop!
        for g in pbar:
            pbar.set_description(f'Processing ({g+1})')
            Niter         = g

            if debug:
                self.verbose=0
            if verbose:
                print(f' SOLVE : Generation == No.{g} ==')

            # update BEST and runHGS
            best  = self.iteration(Niter,best)
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

            for idx, p in self.pop.items():
                if verbose:
                    gtext = 'UPDATE PART POINT'
                    self.declare(gtext)

                self.updateParticle()
            
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

            with open(f'{log_path}{self.pso_name}c1log.pkl','wb') as fid:
                pickle.dump(self.logc1,fid)

            with open(f'{log_path}{self.pso_name}c2log.pkl','wb') as fid:
                pickle.dump(self.logc2,fid)            

            with open(f'{log_path}{self.pso_name}iwlog.pkl','wb') as fid:
                pickle.dump(self.logiw,fid)

            with open(f'{log_path}{self.pso_name}flog.pkl','wb') as fid:
                pickle.dump(self.logf,fid)

            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break

            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)

        # }}}

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

    def read_tem(self, replacements): # {{{
        with open(self.tem_path, 'r') as file:
            template_cont = file.read()

        for key, value in replacements.items():
            template_cont = template_cont.replace(f'# {key} #', str(value))

        return template_cont # }}}

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

    def InitToolBox(self): # {{{
        # Toolbox
        toolbox = base.Toolbox()
        toolbox.register("particle", tools.initIterate, self.generate_solutes)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.particle)
        toolbox.register("update_particle", self.updateParticle)
        toolbox.register("evaluate", self.runHGS)

        return toolbox  # }}}

    def EvolutionaryStateEstimation(self): # {{{

        # step 0: orgarnize particle and regulization
        d = np.zeros([self.npop,0])
        for j in self.par:
            key = self.pop[0][j].keys()
            lloc = np.zeros([self.npop, len(key)])
            for s in self.source:
                for i in range(self.npop):
                    lloc[i] = self.pop[i][j].loc[s]
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
        #print(f' d : {d}')
        # step 1: calculate avg distance of all particle pair
        d1 = np.zeros(self.npop)
        for i in range(self.npop):
            f1 = np.sum((d[i] - d) ** 2, axis=1)
            #print(f' f1 : {f1}')
            f2 = np.sqrt(f1)
            #print(f' f2 : {f2}')
            f3 = np.sum(f2)
            #print(f' f3 : {f3}')
            d1[i] = f3/(self.npop - 1)
        #print(f' d1 : {d1}')
        # step 2: find F & D
        fit = np.zeros(self.npop)
        for i in range(self.npop):
            fit[i] = self.pop[i].fitness.values[0]
        #print(fit)
        fidx = np.argmin(fit)
        #print(fidx)
        dmax = d1[:self.npop].max()
        dmin = d1[:self.npop].min()
        dg   = d1[fidx]
        if self.verbose:
            print(f' dg, dmin, dmax : {dg} / {dmin} / {dmax}')
        # step 3: calculate f
        f = (dg-dmin)/(dmax-dmin)
        if self.verbose:
            print(f' F value: {f}')

        # step 4: define state
        # Case (a)—Exploration
        if 0.0<=f<=0.4:
            uS1 = 0.0
        elif 0.4<f<=0.6:
            uS1 = 5*f - 2
        elif 0.6<f<=0.7:
            uS1 = 1.0
        elif 0.7<f<=0.8:
            uS1 = -10*f + 8
        elif 0.8<f<=1.0:
            uS1 = 0.0
        # Case (b)—Exploitation
        if 0.0<=f<=0.2:
            uS2 = 0
        elif 0.2<f<=0.3:
            uS2 = 10*f - 2
        elif 0.3<f<=0.4:
            uS2 = 1.0
        elif 0.4<f<=0.6:
            uS2 = -5*f + 3
        elif 0.6<f<=1.0:
            uS2 = 0.0
        # Case (c)—Convergence
        if 0.0<=f<=0.1:
            uS3 = 1.0
        elif 0.1<f<=0.3:
            uS3 = -5*f + 1.5
        elif 0.3<f<=1.0:
            uS3 = 0.0
        # Case (d)—Jumping Out
        if 0.0<=f<=0.7:
            uS4 = 0.0
        elif 0.7<f<=0.9:
            uS4 = 5*f - 3.5
        elif 0.9<f<=1.0:
            uS4 = 1.0

        if uS3!=0:
            Current_State = 'S3'
            if uS2!=0:
                Current_State = 'S3&S2'
        elif uS2!=0:
            Current_State = 'S2'
            if uS1!=0:
                Current_State = 'S2&S1'
        elif uS1!=0:
            Current_State = 'S1'
            if uS4!=0:
                Current_State = 'S1&S4'
        elif uS4!=0:
            Current_State = 'S4'

        if self.verbose:
            print(f' Current_State: {Current_State}')

        fstate = self.rule_base[Current_State][self.Previous_State]

        if self.verbose:
            print(f' Final State : {fstate}')
        self.Previous_State = fstate

        # step 5: calculate accerelate coefficient
        delta = np.random.uniform(low=0.05, high=0.1, size=2)
        
        if fstate=='S1': # Exploration
            self.c1 = self.c1 + delta[0]
            self.c2 = self.c2 - delta[1]
        elif fstate=='S2': # Exploitation
            self.c1 = self.c1 + 0.5*delta[0]
            self.c2 = self.c2 - 0.5*delta[1]
        elif fstate=='S3': # Convergence
            self.c1 = self.c1 + 0.5*delta[0]
            self.c2 = self.c2 + 0.5*delta[1]
        elif fstate=='S4': # Jumping Out
            self.c1 = self.c1 - delta[0]
            self.c2 = self.c2 + delta[1]
            
        self.c1 = np.clip(self.c1, 1.5, 2.5)
        self.c2 = np.clip(self.c2, 1.5, 2.5)

        if (3.0<=self.c1+self.c2<=4.0)==False:
            self.c1 = 4.0 * self.c1/(self.c1+self.c2)
            self.c2 = 4.0 * self.c2/(self.c1+self.c2)

        # step 6: calculate inertia weight
        self.w = 1/(1+1.5*np.exp(-2.6*f))
        self.w = np.clip(self.w, self.w_min, self.w_max)

        return f # }}}

    def indexing(self): # {{{
        # Based on 'loc' parameter indexing
        j    = 'loc'
        for i in range(self.npop):
            if self.verbose:
                print(f'before:{self.pop[i][j]}')
            gloc = np.zeros([len(self.source), 3])
            lloc = np.zeros([len(self.source), 3]) 
            # get gbest loc
            for sidx, s in enumerate(self.source):
                gloc[sidx] = self.best[j].loc[s]
            # get each particle loc
            for sidx, s in enumerate(self.source):
                lloc[sidx] = self.pop[i][j].loc[s]
            
            d1 = []
            # calculate distance
            for g in gloc:
                for l in lloc:
                    d = np.sqrt(np.sum((g-l) ** 2))
                    d1.append(d)
            # reshape for find the min value index
            d2 = np.array(d1).reshape(len(gloc), len(lloc))
            if self.verbose:
                print(f'd2 : {d2}')
            row, cidx = linear_sum_assignment(d2)
            cidx2 = np.argmin(d2, axis=0)
            if self.verbose:
                print(f'cidx: {cidx}')
                print(f'cidx2 : {cidx2}')
            ns = [self.source[idx] for idx in cidx]

            for key in self.pop[i].keys():
                origin = self.pop[i][key].copy()
                for nidx, oidx in zip(ns, self.source):
                    self.pop[i][key].loc[nidx] = origin.loc[oidx]
            if self.verbose:
                print(f'after:{self.pop[i][j]}') # }}}
