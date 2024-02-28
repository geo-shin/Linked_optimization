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

class HgsPSO:  # {{{
    def __init__(self, prefix, sources_limits, time_list, obs,
            seed:int       =45,
            model_dir:str  ='base_model', 
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
        self.sources_limits = sources_limits
        self.npop           = npop
        self.sources_limits = sources_limits
        self.nsource        = len(sources_limits)
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
        
        # set up chunk_size
        self.chunk_size     = chunk_size

        # Set up grid size for location
        self.grid_size      = grid_size

        # initialize population
        self.pop            = None

        # initialize creator
        self.creator        = creator
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Particle", dict, 
            fitness=creator.FitnessMin,
            index=None, # index of particle!
            best=None, # global best
            speed_location=list, speed_flux=list,
            smin=None, smax=None,
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

    def generate_solutes(self,re:bool=0): # {{{
        '''
        Usage 
        solutes = self.generate_solutes()
        '''
        # Save the current random state
        cur_state = random.getstate()

        verbose = self.verbose
        if verbose: 
            gtext = 'GENERATE_SOLUTE PART START POINT'
            self.declare(gtext)

        # Set random seed
        random.seed(self.seed)
        if re:
            self.ind_best  = [p.best for p in self.pop]  # 각 particle의 현재 best 값을 저장
        self.pop       = [] # none list
        for idx in range(self.npop):
            # initialize variables
            location           = [] # x, y, z
            flux               = [] # flux
            hk                 = []
            LD                 = []
            speed_location     = []
            speed_flux         = []
            speed_hk           = []
            speed_LD           = []

            for j in range(self.nsource): # time step problem
                pmin, pmax, smin, smax = self.sources_limits[j] # first column
                if verbose:
                    print(f" Generate_solutes : \n {pmin} & {pmax}\n")
                location.append(
                        [random.uniform(min_val, max_val) for min_val, max_val in zip(pmin[:3], pmax[:3])])
                flux.append([random.uniform(pmin[3], pmax[3]) for _ in range(self.ntime)])
                hk.append([random.uniform(pmin[4],pmax[4])])
                LD.append([random.uniform(pmin[5],pmax[5])])
                speed_location.append(
                        [random.uniform(min_val, max_val) for min_val, max_val in zip(smin[:3], smax[:3])])
                speed_flux.append([random.uniform(smin[3], smax[3]) for _ in range(self.ntime)])
                speed_hk.append([random.uniform(smin[4], smax[4])])
                speed_LD.append([random.uniform(smin[5], smax[5])])
            if verbose:
                print(f" location     : {location}")
                print(f" flux         : {flux}")
                print(f" Conductivity : {hk}\n")
                print(f" Dispersivity : {LD}\n")
            particle = creator.Particle({'location':location, 'flux':flux, 'hk':hk, 'LD':LD})
            #    'speed_location':speed_location, 'speed_flux':speed_flux})
            particle.speed_location = speed_location #['speed_location'] = []
            particle.speed_flux     = speed_flux     #['speed_flux']     = []
            particle.speed_hk       = speed_hk
            particle.speed_LD       = speed_LD
            particle.index          = idx
            if re:
                particle.best = self.ind_best[idx]
            else:
                particle.best = None
                
            # stack particle!
            self.pop.append(copy.deepcopy(particle))
        random.setstate(cur_state)
        if verbose:
            gtext = 'GENERATE_SOLUTE PART END POINT'
            self.declare(gtext)

        # }}}

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

                num_neighbors = int(0.1 * n)  # Calculate 10% of the population size
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
                # Generate a list of possible neighbor indices (excluding the particle itself)
                p_neighbors = [i for i in range(self.npop) if i != idx]
                # Randomly select two neighbors
                s_neighbors = random.sample(p_neighbors, 2)
                # Save selected neighbors' index
                p.neighbors = s_neighbors
                if verbose:
                    print(f"selected Neighbor : {s_neighbors} ")
        else:
            pass

        # }}}
  
    def splitpop(self,chunk_size:int=None):# {{{
        '''
        split list!
        '''
        if chunk_size == None:
            chunk_size = self.chunk_size
            
        new_list = []
        for i in range(0,len(self.pop), chunk_size):
            new_list.append(self.pop[i:i+chunk_size])
        return new_list
        # }}}

    def updateParticle(self, part, best, phi1, phi2, iw:int=None, verbose:bool=0): # {{{
        '''
        Explain 
        Update each particle's positions.
        '''

        limit = self.sources_limits
        nsource = len(part['location'])

        if verbose:
            gtext = 'UPDATE_PARTICLE PART START POINT'
            self.declare(gtext)

        if verbose:
            for j in range(nsource):
                limits = limit[j]
                print(f" Generate_solutes : \n {limits} \n")

        '''
        loc  - current location
        locl - local best location
        locg - global best location
        speedg  - local best speed
        '''
        old_part    = copy.deepcopy(part)
        old_pb      = copy.deepcopy(part.best)
        old_best    = copy.deepcopy(best)


        if verbose:
            print(f'updateParticle : check idx = {part.index}')

        for loc_idx in range(nsource):
            pmin, pmax, smin, smax = limit[loc_idx]
            loc   = part['location'][loc_idx]
            locp  = part.best['location'][loc_idx]
            locg  = best['location'][loc_idx]
            
            if verbose:
                print('\nupdateParticle : check current location')
                print(f'   loc_idx : {loc_idx}')
                print(f'   locp    : {locp}')
                print(f'   loc     : {loc}')
                print(f'   locg    : {locg}')

            u1_location = [random.uniform(0, phi1) for _ in loc]
            u2_location = [random.uniform(0, phi2) for _ in loc]
            v_u1_location = [a * (b - c) for a, b, c in zip(u1_location, locp, loc)]
            v_u2_location = [a * (b - c) for a, b, c in zip(u2_location, locg, loc)]
            
            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_location   : {u1_location}')
                print(f'   u2_location   : {u2_location}')               
                print(f'   v_u1_location : {v_u1_location}')
                print(f'   v_u2_location : {v_u2_location}')
                
            if iw is not None: # Iw version
                part.speed_location[loc_idx] = [iw * a + b + c for a, b, c in zip(part.speed_location[loc_idx], v_u1_location, v_u2_location)]
            else:
                part.speed_location[loc_idx] = [a + b + c for a, b, c in zip(part.speed_location[loc_idx], v_u1_location, v_u2_location)]
				
            if verbose:
                print(f'\nupdateParticle : check location speed \n{part.speed_location[loc_idx]} ')

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
            flux    = part['flux'][t_idx]
            fluxp   = part.best['flux'][t_idx]
            fluxg   = best['flux'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" flux_idx   : {t_idx}")
                print(f' flux       : {flux}')
                print(f' fluxp      : {fluxp}')
                print(f' fluxg      : {fluxg}')
            u1_flux = [random.uniform(0, phi1) for _ in flux]
            u2_flux = [random.uniform(0, phi2) for _ in flux]
            v_u1_flux = [a * (b - c) for a, b, c in zip(u1_flux, fluxp, flux)]
            v_u2_flux = [a * (b - c) for a, b, c in zip(u2_flux, fluxg, flux)]
            
            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_flux   : {u1_flux}')
                print(f'   u2_flux   : {u2_flux}')
                print(f'   v_u1_flux : {v_u1_flux}')
                print(f'   v_u2_flux : {v_u2_flux}')
            
            if iw:            
                part.speed_flux[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_flux[t_idx], v_u1_flux, v_u2_flux)]
            else:
                part.speed_flux[t_idx] = [a + b + c for a, b, c in zip(part.speed_flux[t_idx], v_u1_flux, v_u2_flux)]
            
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_flux[t_idx]):
                if abs(speed) > smax[3]:
                    part.speed_flux[t_idx][i] = math.copysign(smax[3], speed)
            if verbose:
                print(f'\nupdateParticle : check flux speed \n{part.speed_flux[t_idx]}')
            
            # Update local flux , speed
            part['flux'][t_idx] = [round(a + b, 10) for a, b in zip(part['flux'][t_idx], part.speed_flux[t_idx])]
            
            for i, flux in enumerate(part['flux'][t_idx]):
                if flux < pmin[3]:
                    part['flux'][t_idx][i] = pmin[3]
                elif flux > pmax[3]:
                    part['flux'][t_idx][i] = pmax[3]

        # K update
        for t_idx in range(len(part['hk'])):
            pmin, pmax, smin, smax = limit[t_idx]
            hk    = part['hk'][t_idx]
            hkp   = part.best['hk'][t_idx]
            hkg   = best['hk'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' hk       : {hk}')
                print(f' hkp      : {hkp}')
                print(f' hkg      : {hkg}')
            u1_hk = [random.uniform(0, phi1) for _ in hk]
            u2_hk = [random.uniform(0, phi2) for _ in hk]
            v_u1_hk = [a * (b - c) for a, b, c in zip(u1_hk, hkp, hk)]
            v_u2_hk = [a * (b - c) for a, b, c in zip(u2_hk, hkg, hk)]
            
            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_hk   : {u1_hk}')
                print(f'   u2_hk   : {u2_hk}')
                print(f'   v_u1_hk : {v_u1_hk}')
                print(f'   v_u2_hk : {v_u2_hk}')
            
            if iw:            
                part.speed_hk[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_hk[t_idx], v_u1_hk, v_u2_hk)]
            else:
                part.speed_hk[t_idx] = [a + b + c for a, b, c in zip(part.speed_hk[t_idx], v_u1_hk, v_u2_hk)]
            
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_hk[t_idx]):
                if abs(speed) > smax[4]:
                    part.speed_hk[t_idx][i] = math.copysign(smax[4], speed)
            if verbose:
                print(f'\nupdateParticle : check K speed \n{part.speed_hk[t_idx]}')
            
            # Update local flux , speed
            part['hk'][t_idx] = [round(a + b, 10) for a, b in zip(part['hk'][t_idx], part.speed_hk[t_idx])]
            
            for i, hk in enumerate(part['hk'][t_idx]):
                if hk < pmin[4]:
                    part['hk'][t_idx][i] = pmin[4]
                elif hk > pmax[4]:
                    part['hk'][t_idx][i] = pmax[4]

        # LD update
        for t_idx in range(len(part['LD'])):
            pmin, pmax, smin, smax = limit[t_idx]
            LD    = part['LD'][t_idx]
            LDp   = part.best['LD'][t_idx]
            LDg   = best['LD'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' LD       : {LD}')
                print(f' LDp      : {LDp}')
                print(f' LDg      : {LDg}')
            u1_LD = [random.uniform(0, phi1) for _ in LD]
            u2_LD = [random.uniform(0, phi2) for _ in LD]
            v_u1_LD = [a * (b - c) for a, b, c in zip(u1_LD, LDp, LD)]
            v_u2_LD = [a * (b - c) for a, b, c in zip(u2_LD, LDg, LD)]
            
            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_LD   : {u1_LD}')
                print(f'   u2_LD   : {u2_LD}')
                print(f'   v_u1_LD : {v_u1_LD}')
                print(f'   v_u2_LD : {v_u2_LD}')
            
            if iw:            
                part.speed_LD[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_LD[t_idx], v_u1_LD, v_u2_LD)]
            else:
                part.speed_LD[t_idx] = [a + b + c for a, b, c in zip(part.speed_LD[t_idx], v_u1_LD, v_u2_LD)]
            
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_LD[t_idx]):
                if abs(speed) > smax[5]:
                    part.speed_LD[t_idx][i] = math.copysign(smax[5], speed)
            if verbose:
                print(f'\nupdateParticle : check K speed \n{part.speed_LD[t_idx]}')
            
            # Update local flux , speed
            part['LD'][t_idx] = [round(a + b, 10) for a, b in zip(part['LD'][t_idx], part.speed_LD[t_idx])]
            
            for i, LD in enumerate(part['LD'][t_idx]):
                if LD < pmin[5]:
                    part['LD'][t_idx][i] = pmin[5]
                elif LD > pmax[5]:
                    part['LD'][t_idx][i] = pmax[5]
       
        if verbose:
            print('\nupdateParticle: check update particle.')
            print('\nlocation_before :',old_part['location'],'\nspeed :',old_part.speed_location)
            print('location_after  :',part['location'],'\nspeed :',part.speed_location)
            print('\nflux_before     :',old_part['flux'],'\nspeed :',old_part.speed_flux)
            print('flux_after      :',part['flux'],'\nspeed :',part.speed_flux)
            print('\nK before        :',old_part['hk'],'\nspeed :',old_part.speed_hk)
            print('hk after        :',part['hk'],'\nspeed :',part.speed_hk)
            print('\nLD before       :',old_part['LD'],'\nspeed :',old_part.speed_LD)
            print('LD_after        :',part['LD'],'\nspeed :',part.speed_LD)
            gtext = 'UPDATE_PARTICLE PART END POINT'
            self.declare(gtext)

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

    def runHGS(self,population): # {{{
        # initialize variables
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
            gtext = 'RUNHGS PART START POINT'
            self.declare(gtext)

        # make destination directory.
        if not os.path.exists(dest_dir):
            print(f'make directory {dest_dir}')
            shutil.copytree(f'{model_dir}', dest_dir,dirs_exist_ok=True)
        else:
            command = f'cd {dest_dir} && ./clearall.sh'
            os.system(command)
            pass

        if verbose: print('runHGS: Write contaminant_list information')  # {{{
        CNum_string     = ''
        Cinclude_string = ''
        C_lines         = ''
        
        # New string for initial concentration
        Cinit_string = "choose nodes all\ninitial concentration\n"
        for p in population:
            pindex = p.index
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

        for idx, p in zip(range(local_npop),population):
            pindex        = p.index # global index in array
            fluxarray     = np.zeros((local_npop,))
            Cfname = f'./01_inc/Contaminant_{pindex}.inc'
            # for time step iteration
            C_lines = ''
            for location, flux_ts in zip(p['location'], p['flux']):
                # write location
                C_lines  += f'''\nclear chosen nodes
Choose node
{location[0]} {location[1]} 30.5

create node set
Source_point1

Specified mass flux
{ntime}\n'''

                # Write sources with the same flux value for each timestep
                fluxarray[:] = 0.0
                for time, flux in zip(timelist, flux_ts):
                    fluxarray[idx] = flux
                    C_lines += f"{time[0]} {time[1]} "
                    for f in fluxarray:
                        C_lines += f' {f}'
                    C_lines += '\n'
  
            # set contamination name
            with open(os.path.join(dest_dir,Cfname),'w') as fid:
                fid.write(C_lines)
        # }}}

        if verbose: print('runHGS: Write K') # {{{
            
        for idx, p in zip(range(local_npop),population):
            k      = np.mean(p['hk'])
            LD     = np.mean(p['LD'])
            kfname = f'./{prefix}.mprops'
            k_lines = f'''!AL
Al
!-- flow properties --
k isotropic
1.0e-4

porosity
0.25

!-- Transport properties --

longitudinal dispersivity
40
transverse dispersivity
9.6
vertical transverse dispersivity
1
tortuosity
1.0

end mat


Fr
!-- flow properties --
k isotropic
{k}
specific storage
0.0                      
porosity
0.05   

!-- Transport properties --

longitudinal dispersivity
{LD}
transverse dispersivity
{LD*0.1}
vertical transverse dispersivity
{LD*0.1}


end mat'''
            with open(os.path.join(dest_dir,kfname),'w') as fid:
                fid.write(k_lines)
                # }}}

        retry_count = 0
        max_retries = 5

        while retry_count <= max_retries:
            if verbose: 
                print('runHGS: Run HGS.') # {{{

            command = f'cd {dest_dir} && {exe_grok} > /dev/null'
            os.system(command)
            command = f'cd {dest_dir} && {exe_hgs} > /dev/null'
            os.system(command)
            # }}}

            try:
                if verbose: 
                    print(f'runHGS: Get results') # {{{
                total_cost = np.zeros((local_npop,))
                for idx, p in zip(range(len(population)), population):
                    pindex = p.index
                    for well_name in obs.keys():
                        obs_time = obs[well_name]['time']
                        obs_conc = obs[well_name]['conc']

                        # 해당 입자, 용액, 및 well_name에 대한 파일 패턴 찾기
                        fname = os.path.join(dest_dir, f'{prefix}o.observation_well_conc.{well_name}.PCE{pindex}.dat')
                        if not os.path.isfile(fname):
                            raise Exception(f'ERROR: we cannot find {fname}')
                            
                        model_ts   = np.loadtxt(fname, skiprows=25)
                        model_time = model_ts[:,0]
                        model_conc = model_ts[:,1]

                        for otime, oconc in zip(obs_time, obs_conc):
                            pos  = (otime == model_time)
                            mconc = model_conc[pos] # find specific time concentration.
                            # calculate concentration.
                            total_cost[idx] += (mconc-oconc)**2
                break  # If there's no error, break out of the while loop
            except:
                if retry_count == max_retries:
                    if verbose:
                        print('runHGS: Max retries reached. Stopping...')
                    break
                if verbose: 
                    print('runHGS: An error occurred. Retrying...')
                retry_count += 1
        # }}}
        
        if verbose: print('runHGS: assign all cost in each poplulation')
        for p, cost in zip(population, total_cost):
            pindex = p.index
            self.pop[pindex].fitness.values = (cost,)

        if verbose:
            gtext = 'RUNHGS PART END POINT'
            self.declare(gtext)
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

        # Variable setting
        start_time     = time.time()
        toolbox        = self.InitToolBox()
        best           = None
        Nbest          = math.inf
        i              = 0
        LogPosition    = {}
        LogPosition[0] = self.WritePopDict(self.pop, best)
        p1_1,p1_2      = phi1, phi2

        # Main loop!
        for g in range(self.generation):
            Niter         = g
            self.ind_best = []
            self.gl_best  = []

            if debug:
                self.verbose=0

            # HGS model constructed
            best  = self.iteration(Niter,best)
            for idx, p in enumerate(self.pop):
                # original version
                p, p.best, best = self.updateParticle(p, best,p1_1,p1_2, verbose=self.verbose)
            
            if debug:
                self.verbose=1
            # Check the updated best particle information.
            if verbose:
                print('update finished !\n')
                if best is not None:
                    print(f'global best after updated : {best}\n')
                    print(f'global best value         : {best.fitness.values} \n')
            LogPosition[g+1] = self.WritePopDict(self.pop, best)
            
            # Time consume
            end_time = time.time()
            execution_time = end_time - start_time
            print("\n-----Execution time:----- ", execution_time, "seconds")
            
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
 
        return LogPosition, execution_time
        # }}}

    def WritePopDict(self,pop,best): # {{{
        global_best  = None
        global_value = None
        pop          = None
        if best is not None:
            global_best  = dict(best)
            global_value = best.fitness.values[0]
            pop = [dict(p,value=p.fitness.values[0],
            index=p.index,speed_location=p.speed_location, speed_flux=p.speed_flux) for p in self.pop]

        result = dict(pop=copy.deepcopy(pop),
                    global_best  = global_best,
                    global_value = global_value,
                    )

        return result
    # }}}
    
    def iteration(self,Niter:int,best,verbose:bool=None): # {{{
        verbose   = self.verbose
        g         = Niter
        box_width = 60 
        gen_text  = f"Generation {g + 1} of {self.generation}".center(box_width - 12)
        # show iteration box
        box = (
            f"\n{'@' * box_width}"
            f"\n{'@' * 6}{' ' * (box_width - 12)}{'@' * 6}"
            f"\n{'@' * 6}{gen_text}{'@' * 6}"
            f"\n{'@' * 6}{' ' * (box_width - 12)}{'@' * 6}"
            f"\n{'@' * box_width}\n"
        )
        print(box)

        start_gen = time.time()

        if verbose:
            gtext = 'ITERATION PART START POINT'
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
        for p in self.pop:
            if verbose:
                print(f'iteration : check idx = {p.index}\n')
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
            gtext = 'ITERATION PART END POINT'
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

class HgsAPSO(HgsPSO):
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

        # this is main loop!
        for g in range(self.generation):
            Niter = g
            best  = self.iteration(Niter,best)
            iw    = self.inertia_weight(Niter)
            p1_1 , p1_2 = self.Adaptive_phi(Niter, p1_1, p1_2)
            for idx, p in enumerate(self.pop):
                # Adpative version
                p, p.best, best = self.updateParticle(p, best,p1_1,p1_2, verbose=verbose)
                # iw version
                # p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,iw,verbose=verbose)
                # IWAdaptive version
                #p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,iw,verbose=verbose)
            
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
            
            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break

            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)

            if verbose:
                gtext = 'SOLVE PART END POINT'
                self.declare(gtext)

        return LogPosition, execution_time
        # }}}

class HgsCFPSO(HgsPSO):   
    def updateParticle(self, part, best, phi1, phi2, iw:int=None, verbose:bool=0): # {{{

        limit = self.sources_limits
        nsource = len(part['location']) 
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
        # K update
        for t_idx in range(len(part['hk'])):
            pmin, pmax, smin, smax = limit[t_idx]
            hk    = part['hk'][t_idx]
            hkp   = part.best['hk'][t_idx]
            hkg   = best['hk'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' hk       : {hk}')
                print(f' hkp      : {hkp}')
                print(f' hkg      : {hkg}')
            u1_hk = [random.uniform(0, phi1) for _ in hk]
            u2_hk = [random.uniform(0, phi2) for _ in hk]
            v_u1_hk = [a * (b - c) for a, b, c in zip(u1_hk, hkp, hk)]
            v_u2_hk = [a * (b - c) for a, b, c in zip(u2_hk, hkg, hk)]
            
            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_hk   : {u1_hk}')
                print(f'   u2_hk   : {u2_hk}')
                print(f'   v_u1_hk : {v_u1_hk}')
                print(f'   v_u2_hk : {v_u2_hk}')
            
            if iw:            
                part.speed_hk[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_hk[t_idx], v_u1_hk, v_u2_hk)]
            else:
                part.speed_hk[t_idx] = [a + b + c for a, b, c in zip(part.speed_hk[t_idx], v_u1_hk, v_u2_hk)]
                
            if verbose:
                print(f'speed_hk before calculation : \n{part.speed_hk[t_idx]}')
            # Constriction Factor Calculation
            CF_speed_hk = [K*speed for speed in part.speed_hk[t_idx]]
            part.speed_hk[t_idx] = CF_speed_hk
            if verbose:
            	print(f'speed_hk after calculation : \n{part.speed_hk[t_idx]}')
            
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_hk[t_idx]):
                if abs(speed) > smax[4]:
                    part.speed_hk[t_idx][i] = math.copysign(smax[4], speed)
            if verbose:
                print(f'\nupdateParticle : check K speed \n{part.speed_hk[t_idx]}')
            
            # Update local flux , speed
            part['hk'][t_idx] = [round(a + b, 10) for a, b in zip(part['hk'][t_idx], part.speed_hk[t_idx])]
            
            for i, hk in enumerate(part['hk'][t_idx]):
                if hk < pmin[4]:
                    part['hk'][t_idx][i] = pmin[4]
                elif hk > pmax[4]:
                    part['hk'][t_idx][i] = pmax[4]        

        # K update
        for t_idx in range(len(part['LD'])):
            pmin, pmax, smin, smax = limit[t_idx]
            LD    = part['LD'][t_idx]
            LDp   = part.best['LD'][t_idx]
            LDg   = best['LD'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' LD       : {LD}')
                print(f' LDp      : {LDp}')
                print(f' LDg      : {LDg}')
            u1_LD = [random.uniform(0, phi1) for _ in LD]
            u2_LD = [random.uniform(0, phi2) for _ in LD]
            v_u1_LD = [a * (b - c) for a, b, c in zip(u1_LD, LDp, LD)]
            v_u2_LD = [a * (b - c) for a, b, c in zip(u2_LD, LDg, LD)]
            
            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_LD   : {u1_LD}')
                print(f'   u2_LD   : {u2_LD}')
                print(f'   v_u1_LD : {v_u1_LD}')
                print(f'   v_u2_LD : {v_u2_LD}')
            
            if iw:            
                part.speed_LD[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_LD[t_idx], v_u1_LD, v_u2_LD)]
            else:
                part.speed_LD[t_idx] = [a + b + c for a, b, c in zip(part.speed_LD[t_idx], v_u1_LD, v_u2_LD)]
                
            if verbose:
                print(f'speed_LD before calculation : \n{part.speed_LD[t_idx]}')
            # Constriction Factor Calculation
            CF_speed_LD = [K*speed for speed in part.speed_LD[t_idx]]
            part.speed_LD[t_idx] = CF_speed_LD
            if verbose:
            	print(f'speed_LD after calculation : \n{part.speed_LD[t_idx]}')
            
            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_LD[t_idx]):
                if abs(speed) > smax[5]:
                    part.speed_LD[t_idx][i] = math.copysign(smax[5], speed)
            if verbose:
                print(f'\nupdateParticle : check K speed \n{part.speed_LD[t_idx]}')
            
            # Update local flux , speed
            part['LD'][t_idx] = [round(a + b, 10) for a, b in zip(part['LD'][t_idx], part.speed_LD[t_idx])]
            
            for i, LD in enumerate(part['LD'][t_idx]):
                if LD < pmin[5]:
                    part['LD'][t_idx][i] = pmin[5]
                elif LD > pmax[5]:
                    part['LD'][t_idx][i] = pmax[5]    
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

class HgsIWPSO(HgsPSO):
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

        # this is main loop!
        for g in range(self.generation):
            Niter = g
            best  = self.iteration(Niter,best)
            iw    = self.inertia_weight(Niter)
            #p1_1 , p1_2 = self.Adaptive_phi(Niter, p1_1, p1_2)
            for idx, p in enumerate(self.pop):
                # None iw version
                #p, p.best, best = self.updateParticle(p, best,p1_1,p1_2, verbose=verbose)
                # iw version
                p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,iw,verbose=verbose)
                # Adaptive version
                #p, p.best, best = self.updateParticle(p, best,p1_1,p1_2,iw,verbose=verbose)
            
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
            
            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break

            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)

            if verbose:
                gtext = 'SOLVE PART END POINT'
                self.declare(gtext)

        return LogPosition, execution_time
        # }}}

class HgsAIWPSO(HgsPSO):
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

        # this is main loop!
        for g in range(self.generation):
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
            
            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break
 
            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)

            if verbose:
                gtext = 'SOLVE PART END POINT'
                self.declare(gtext)

        return LogPosition, execution_time
        # }}}

class HgsBBPSO(HgsPSO):
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
                    
        # K update
        for t_idx in range(len(part['hk'])):
            pmin, pmax, smin, smax = limit[t_idx]
            hk    = part['hk'][t_idx]
            hkp   = part.best['hk'][t_idx]
            hkg   = best['hk'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' hk       : {hk}')
                print(f' hkp      : {hkp}')
                print(f' hkg      : {hkg}')
                
            if verbose:
                print(f"hk          : {hk}")
            # 새로운 위치 샘플링 방식 도입
            mean_hk = [(hkp[i] + hkg[i]) / 2 for i in range(len(hk))]
            std_dev_hk = [(hkp[i] - hkg[i]) / 2 for i in range(len(hk))]
            part['hk'][t_idx] = [round(random.gauss(mean_hk[i], std_dev_hk[i]),10) for i in range(len(hk))]

            if verbose:
                print(f" update_hk : {mean_hk}")
                print(f" std_hk : {std_dev_hk}")
            
            for i, hk in enumerate(part['hk'][t_idx]):
                if hk < pmin[4]:
                    part['hk'][t_idx][i] = pmin[4]
                elif hk > pmax[4]:
                    part['hk'][t_idx][i] = pmax[4]   
        
        # K update
        for t_idx in range(len(part['LD'])):
            pmin, pmax, smin, smax = limit[t_idx]
            LD    = part['LD'][t_idx]
            LDp   = part.best['LD'][t_idx]
            LDg   = best['LD'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' LD       : {LD}')
                print(f' LDp      : {LDp}')
                print(f' LDg      : {LDg}')
                
            if verbose:
                print(f"LD          : {LD}")
            # 새로운 위치 샘플링 방식 도입
            mean_LD = [(LDp[i] + LDg[i]) / 2 for i in range(len(LD))]
            std_dev_LD = [(LDp[i] - LDg[i]) / 2 for i in range(len(LD))]
            part['LD'][t_idx] = [round(random.gauss(mean_LD[i], std_dev_LD[i]),10) for i in range(len(LD))]

            if verbose:
                print(f" update_LD : {mean_LD}")
                print(f" std_LD : {std_dev_LD}")
            
            for i, LD in enumerate(part['LD'][t_idx]):
                if LD < pmin[5]:
                    part['LD'][t_idx][i] = pmin[5]
                elif LD > pmax[5]:
                    part['LD'][t_idx][i] = pmax[5]             
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

class HgsCLPSO(HgsPSO):
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
         # K update
        for t_idx in range(len(part['hk'])):
            pmin, pmax, smin, smax = limit[t_idx]
            hk    = part['hk'][t_idx]
            hkp   = part.best['hk'][t_idx]
            hkg   = best['hk'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' hk       : {hk}')
                print(f' hkp      : {hkp}')
                print(f' hkg      : {hkg}')

            updated_hkl = []
            for i in range(len(hk)):
                if random.random() < Pc:
                    neighbor_best = min([self.pop[idx] for idx in part.neighbors], key=lambda x: x.best.fitness)
                    selected_neighbor = neighbor_best.index
                    updated_hkl.append(self.pop[selected_neighbor].best['hk'][t_idx][i])
                    if verbose:
                        print(f"Selected Neighbor Index: {neighbor_best.index}")
                        print(f"Updated Location: {updated_hkl[-1]}")
                        print(f"Current Updated Location List: {updated_hkl}")
                else:
                    updated_hkl.append(part.best['hk'][t_idx][i])
                    if verbose:
                        print("particle best selected")
            hkl  = updated_hkl

            if verbose:
                print(f"hkl         : {hkl}")
                print(f"hk          : {hk}")  

            u1_hk = [random.uniform(0, phi1) for _ in hk]
            u2_hk = [random.uniform(0, phi2) for _ in hk]
            v_u1_hk = [a * (b - c) for a, b, c in zip(u1_hk, hkl, hk)]
            v_u2_hk = [a * (b - c) for a, b, c in zip(u2_hk, hkg, hk)]

            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_hk   : {u1_hk}')
                print(f'   u2_hk   : {u2_hk}')
                print(f'   v_u1_hk : {v_u1_hk}')
                print(f'   v_u2_hk : {v_u2_hk}')

            if iw:            
                part.speed_hk[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_hk[t_idx], v_u1_hk, v_u2_hk)]
            else:
                part.speed_hk[t_idx] = [a + b + c for a, b, c in zip(part.speed_hk[t_idx], v_u1_hk, v_u2_hk)]

            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_hk[t_idx]):
                if abs(speed) > smax[4]:
                    part.speed_hk[t_idx][i] = math.copysign(smax[4], speed)
            if verbose:
                print(f'\nupdateParticle : check K speed \n{part.speed_hk[t_idx]}')

            # Update local flux , speed
            part['hk'][t_idx] = [round(a + b, 10) for a, b in zip(part['hk'][t_idx], part.speed_hk[t_idx])]

            for i, hk in enumerate(part['hk'][t_idx]):
                if hk < pmin[4]:
                    part['hk'][t_idx][i] = pmin[4]
                elif hk > pmax[4]:
                    part['hk'][t_idx][i] = pmax[4]

        # K update
        for t_idx in range(len(part['LD'])):
            pmin, pmax, smin, smax = limit[t_idx]
            LD    = part['LD'][t_idx]
            LDp   = part.best['LD'][t_idx]
            LDg   = best['LD'][t_idx]

            if verbose:
                print(f'\nupdateParticle : check current flux')
                print(f" k_idx    : {t_idx}")
                print(f' LD       : {LD}')
                print(f' LDp      : {LDp}')
                print(f' LDg      : {LDg}')

            updated_LDl = []
            for i in range(len(LD)):
                if random.random() < Pc:
                    neighbor_best = min([self.pop[idx] for idx in part.neighbors], key=lambda x: x.best.fitness)
                    selected_neighbor = neighbor_best.index
                    updated_LDl.append(self.pop[selected_neighbor].best['LD'][t_idx][i])
                    if verbose:
                        print(f"Selected Neighbor Index: {neighbor_best.index}")
                        print(f"Updated Location: {updated_LDl[-1]}")
                        print(f"Current Updated Location List: {updated_LDl}")
                else:
                    updated_LDl.append(part.best['LD'][t_idx][i])
                    if verbose:
                        print("particle best selected")
            LDl  = updated_LDl

            if verbose:
                print(f"LDl         : {LDl}")
                print(f"LD          : {LD}")  

            u1_LD = [random.uniform(0, phi1) for _ in LD]
            u2_LD = [random.uniform(0, phi2) for _ in LD]
            v_u1_LD = [a * (b - c) for a, b, c in zip(u1_LD, LDl, LD)]
            v_u2_LD = [a * (b - c) for a, b, c in zip(u2_LD, LDg, LD)]

            if verbose:
                print('\nupdateParticle : check update speed')
                print(f'   u1_LD   : {u1_LD}')
                print(f'   u2_LD   : {u2_LD}')
                print(f'   v_u1_LD : {v_u1_LD}')
                print(f'   v_u2_LD : {v_u2_LD}')

            if iw:            
                part.speed_LD[t_idx] = [iw * a + b + c for a, b, c in zip(part.speed_LD[t_idx], v_u1_LD, v_u2_LD)]
            else:
                part.speed_LD[t_idx] = [a + b + c for a, b, c in zip(part.speed_LD[t_idx], v_u1_LD, v_u2_LD)]

            # 속도 제한 (flux)
            for i, speed in enumerate(part.speed_LD[t_idx]):
                if abs(speed) > smax[5]:
                    part.speed_LD[t_idx][i] = math.copysign(smax[5], speed)
            if verbose:
                print(f'\nupdateParticle : check K speed \n{part.speed_LD[t_idx]}')

            # Update local flux , speed
            part['LD'][t_idx] = [round(a + b, 10) for a, b in zip(part['LD'][t_idx], part.speed_LD[t_idx])]

            for i, LD in enumerate(part['LD'][t_idx]):
                if LD < pmin[5]:
                    part['LD'][t_idx][i] = pmin[5]
                elif LD > pmax[5]:
                    part['LD'][t_idx][i] = pmax[5]
                    
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

        # this is main loop!
        for g in range(self.generation):
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
            
            # Check the criteria
            if best.fitness.values[0] < Con_min:
                print(' Opimization ended to Criteria')
                break
  
            if Ncrit:
                i ,Nbest = self.regen(Nbest, best, i, Ncrit)

            if verbose:
                gtext = 'SOLVE PART END POINT'
                self.declare(gtext)

        return LogPosition, execution_time
        # }}}
