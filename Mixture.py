# This is a Python port of Joy's affinity maturation flexibility code
# available at: https://github.com/jlouveau/Toy_Model_for_John

import csv
import sys
import os
import importlib
import numpy as np                          # numerical tools
from copy import deepcopy                   # deepcopy copies a data structure without any implicit references
#from scipy.stats import genextreme          # generalized extreme value distribution
from timeit import default_timer as timer   # timer for performance
import importlib

###### Global parameters ######
    
p_mut        = 0.14                             # probability of mutation per division round
p_CDR        = 1.00                             # probability of mutation in the CDR region
p_CDR_lethal = 0.30                             # probability that a CDR mutation is lethal
p_CDR_silent = 0.50                             # probability that a CDR mutation is silent
p_CDR_affect = 1. - p_CDR_lethal - p_CDR_silent # probability that a CDR mutation affects affinity
p_var        = 0.10                             # probability that a CDR mutation affects the variable region in the antigens used in the vaccine
p_cons       = 1.0 - p_var                      # probability that a CDR mutation affects the conserved (constant) region
p_FR_lethal  = 0.80                             # probability that a framework (FR) mutation is lethal
p_FR_silent  = 0.                               # probability that a FR mutation is silent
p_FR_affect  = 1. - p_FR_lethal - p_FR_silent   # probability that a FR mutation affects affinity

length            = 46
consLength        = 18
epsilon           = 1e-16
testpanelSize     = 100
breadth_threshold = 20
alpha             = 0.25
nb_GC_founders    = 10
activation_energy = 9
delta_energy      = 0.50
nb_seeding_cells  = 15
dx_mutation       = 1.0
dx_penalty        = 1.0
h_high            = 1.5
h_low             = -1
nb_trial          = 1000

energy_scale = 0.08            # inverse temperature
E0           = 3.00            # mean binding energy for mixing with flexibility
maxQ          = 1               # max value for Q
minQ          = 1               # min value for Q
sigmaQ       = 0.00            # standard deviation for changes in flexibility with FR mutation
help_cutoff  = 0.70            # only B cells in the top (help_cutoff) fraction of binders receive T cell help
p_recycle    = 0.70            # probability that a B cell is recycled
p_exit       = 1. - p_recycle  # probability that a B cell exits the GC

mu     =  1.9   # lognormal mean
sigma  =  0.5   # lognormal standard deviation
corr   =  0.0   # correlation between antigen variable regions
o      =  3.0   # lognormal offset
mumat  = mu * np.ones(length)
sigmat = sigma * np.diag(np.ones(length))

# Upload dictionary of antigens
import dictionary_little_code
reload(dictionary_little_code)
from dictionary_little_code import dicAgs
from dictionary_little_code import dicconc
from dictionary_little_code import dicGCDur
from dictionary_little_code import flag

#Upload seeding cells
import seedingBcell
reload(seedingBcell)
from seedingBcell import seedingCells
 
def create_test_panel(panelSize):
    varLength=length-consLength
    testPanel = {}
    for i in range(panelSize):
        #testAg = []
        #for k in range(length-consLength): testAg=np.append(testAg,-1).tolist()
        testAg = (np.random.choice([-1, 1], varLength, p=[0.5, 0.5])).tolist()
        for j in range(consLength): testAg=np.append(testAg,1).tolist()
        testPanel.update({i: testAg})
        #print(testPanel)
    return testPanel
       
#Create test panel
testpanel = create_test_panel(testpanelSize) 
    
###### B cell clone class ######

class BCell:

    def __init__(self, nb = 512, **kwargs):
        """ Initialize clone-specific variables. 
            nb          - population size
            res         - array of the values for each residue of the B cell 
            E           - total binding energy
            Ec          - binding energy to the conserved residues
            Q           - overlap parameter, proxy for rigidity (most 0 ---> 1 least flexible)
            nb_FR_mut   - number of accumulated FR mutations
            nb_CDR_mut  - number of accumulated CDR mutations
            antigens    - list of antigens
            breadth     - breadth against test panel
            nb_Ag       - number of antigens available
            last_bound  - number of individuals that last bound each Ag
            mut_res_id  - index of mutated residue
            delta_res   - incremental change at residue
            generation  - generation in the GC reaction
            cycle_number - cycle number
            history     - history of mutations, generation occurred, and effect on Q/Ec and breadth against test panel """
        
        self.nb = nb    # default starting population size = 512 (9 rounds of division)
        
        if 'res' in kwargs:
            self.res  = np.array(kwargs['res']) 
        else:
            print('res not recognized as an input argument')    
            self.res = np.zeros(length) 
    
        if 'E' in kwargs: self.E = kwargs['E']  
        else:             self.E = sum(self.res) #assuming that the initializing Ag equals ones(length)

        if 'Ec' in kwargs: self.Ec = kwargs['Ec']  
        else:              self.Ec = sum(self.res[i] for i in range(length-consLength, length))   
                    
        if 'antigens' in kwargs: self.antigens = np.array(kwargs['antigens'])
        else:                    self.antigens = np.array([np.ones(length)])
        
        if 'breadth' in kwargs: self.breadth = np.array(kwargs['breadth'])
        else:                   self.breadth = 0
            
        if 'nb_Ag' in kwargs:              self.nb_Ag = kwargs['nb_Ag']
        elif np.shape(self.antigens)[0] == length: self.nb_Ag = 1   # assuming that the number of antigens in a cocktail is always smaller than the number of residues
        else:                              self.nb_Ag = np.shape(self.antigens)[0]
                      
        if 'Q' in kwargs: self.Q = kwargs['Q']
        else:             self.Q = maxQ

        if 'nb_FR_mut' in kwargs: self.nb_FR_mut = kwargs['nb_FR_mut']
        else:                     self.nb_FR_mut = 0
        
        if 'nb_CDR_mut' in kwargs: self.nb_CDR_mut = kwargs['nb_CDR_mut']
        else:                      self.nb_CDR_mut = 0
        
        if 'last_bound' in kwargs: self.last_bound = kwargs['last_bound']
        else:                      self.last_bound = np.random.multinomial(self.nb, pvals = [1/float(self.nb_Ag)] * self.nb_Ag)
        
        if 'mut_res_id' in kwargs: self.mut_res_id = kwargs['mut_res_id']
        else:                      self.mut_res_id = length
        
        if 'mut_res_id_his' in kwargs: self.mut_res_id_his = kwargs['mut_res_id_his']
        else:                      self.mut_res_id_his = []

        if 'delta_e' in kwargs: self.delta_e = kwargs['delta_e']
        else:                     self.delta_e = 0

        if 'delta_res' in kwargs: self.delta_res = kwargs['delta_res']
        else:                     self.delta_res = 0

        if 'generation' in kwargs: self.generation = kwargs['generation']
        else:                      self.generation = 0

        if 'cycle_number' in kwargs: self.cycle_number = kwargs['cycle_number']
        else:                        self.cycle_number = 2
            
        if 'history' in kwargs: self.history = kwargs['history']
        else:                   self.history = {'generation' : [self.generation], 'cycle_number' : [self.cycle_number], 'res' : [self.res], 'nb_CDR_mut' : [self.nb_CDR_mut], 'mut_res_id' : [self.mut_res_id], 'E' : [self.E], 'delta_res' : [self.delta_res], 'delta_e': [self.delta_e]}

    """ Return a new copy of the input BCell"""
    @classmethod
    def clone(cls, b):
        return cls(1, res = deepcopy(b.res), antigens = deepcopy(b.antigens), cycle_number = b.cycle_number, nb_Ag = b.nb_Ag, E = b.E, Ec = b.Ec, breadth = b.breadth, Q = b.Q, generation = b.generation, mut_res_id = b.mut_res_id, nb_FR_mut = b.nb_FR_mut, nb_CDR_mut = b.nb_CDR_mut, delta_res = b.delta_res, last_bound = deepcopy(b.last_bound), history = deepcopy(b.history), mut_res_id_his = b.mut_res_id_his, delta_e = b.delta_e)
                   
    def update_history(self):
        """ Add current parameters to the history list. """
        self.history['generation'].append(self.generation)
        self.history['res'].append(self.res)
        self.history['nb_CDR_mut'].append(self.nb_CDR_mut)
        self.history['mut_res_id'] = self.history['mut_res_id'] + [self.mut_res_id]
	#self.history['mut_res_id'].append(self.mut_res_id)
        self.history['E'].append(self.E)      
        self.history['delta_res'].append(self.delta_res)
        self.history['cycle_number'].append(self.cycle_number)
        self.history['delta_e'].append(self.delta_e)
	
    def energy(self, Ag):
        """ Return binding energy with input antigen. """            
        return np.sum(np.multiply(self.res, Ag))

    def conserved_energy(self):
        """ Return binding energy for conserved residues. """
        return sum(self.res[i] for i in range(length-consLength, length))   

    def divide(self, cycle_number):
        """ Run one round of division. """
        self.nb *= 2
        self.generation += 1
        self.cycle_number = cycle_number

    def pick_Antigen(self):
        """ Assuming nb_Ag > 1, return one antigen randomly chosen. """
        return self.antigens[np.random.randint(self.nb_Ag)]

    def calculate_breadth(self, testpanel, threshold, panelSize):   
        test_energies = [self.energy(testpanel[j]) for j in range(testpanelSize)]
        return float(sum(x > threshold for x in test_energies))/panelSize
        
    def update_antigens(self, newAntigens):
        self.antigens = deepcopy(newAntigens)
        if np.shape(newAntigens)[0] == length: self.nb_Ag = 1
        else:                                  self.nb_Ag = np.shape(newAntigens)[0]
    
    def mutate_CDR(self, Ag): ### change parameter of log normal for variable and conserved residues
        """ Change in energy due to affinity-affecting CDR mutation. Only one residue mutates."""
        temp_res1 = deepcopy(self.res)
        temp_cp   = deepcopy(self.res)
        index = np.random.randint(0, length) #randomly chosen residue to be mutated
        self.mut_res_id = index
        self.mut_res_id_his.extend([index])
        delta = (o - np.exp(np.random.normal(mu, sigma)))
        self.delta_e = delta

        if delta > dx_mutation: delta = dx_mutation
        elif delta < -dx_mutation: delta = -dx_mutation        
        self.delta_res = delta * Ag[index]
        temp_res1[index] +=  delta * Ag[index]
        if temp_res1[index] > h_high: temp_res1[index] = h_high
        elif temp_res1[index] < h_low: temp_res1[index] = h_low
        self.nb_CDR_mut += 1
        self.Ec = self.conserved_energy()
        self.breadth = self.calculate_breadth(testpanel, breadth_threshold,testpanelSize)
        self.E = self.energy(Ag)
        self.res = deepcopy(temp_res1)        
        self.update_history()
        
        if (index < length-consLength) and temp_cp[index] != h_high and temp_cp[index] != h_low:
            temp_res2 = deepcopy(self.res)
            indexPenalty = np.random.randint(0, consLength) + (length - consLength)
            self.mut_res_id = indexPenalty 
            deltaPenalty = - alpha * delta
            self.delta_e = deltaPenalty
            if deltaPenalty > dx_penalty: deltaPenalty = dx_penalty
            elif deltaPenalty < -dx_penalty: deltaPenalty = -dx_penalty    
            temp_res2[indexPenalty] +=  deltaPenalty
            if temp_res2[indexPenalty] > h_high: temp_res2[indexPenalty] = h_high
            elif temp_res2[indexPenalty] < h_low: temp_res2[indexPenalty] = h_low
            self.delta_res = deltaPenalty
            self.breadth = self.calculate_breadth(testpanel, breadth_threshold,testpanelSize)
            self.Ec = self.conserved_energy()
            self.E = self.energy(Ag)
            self.res = deepcopy(temp_res2)
            self.mut_res_id_his.extend([indexPenalty])
            self.update_history()

    def mutate_FR(self):
        """ Change in flexibility due to affinity-affecting framework (FR) mutation. """
        dQ = np.random.normal(0, sigmaQ)
        if   self.Q + dQ > maxQ:
            self.Q   = maxQ
        elif self.Q + dQ < minQ:
            self.Q   = minQ
            #self.nb -= 1
        else:
            self.Q = self.Q + dQ
        self.nb_FR_mut += 1
        self.calculate_breadth(testpanel, breadth_threshold,testpanelSize)
        self.update_history()       

    def shm(self,cycle_number):
        """ Run somatic hypermutation and return self + new B cell clones. """
        
        # get number of cells that mutate, remove from the clone
        new_clones = []
        n_mut      = np.random.binomial(self.nb, p_mut)
        self.nb   -= n_mut
            
        # get number of CDR vs framework (FR) mutations
        n_CDR = np.random.binomial(n_mut, p_CDR)
        n_FR  = n_mut - n_CDR
            
        # process CDR mutations
        n_die, n_silent, n_affect  = np.random.multinomial(n_CDR, pvals = [p_CDR_lethal, p_CDR_silent, p_CDR_affect])
        self.nb                   += n_silent #add silent mutations to the parent clone
        for i in range(n_affect):
            b = BCell.clone(self)
            if b.nb_Ag > 1: Ag = b.pick_Antigen()
            else: Ag = b.antigens[0]
            b.mutate_CDR(Ag)
            new_clones.append(b)
        
        # process FR mutations
        n_die, n_silent, n_affect  = np.random.multinomial(n_FR, pvals = [p_FR_lethal, p_FR_silent, p_FR_affect])
        self.nb                   += n_silent
        for i in range(n_affect):
            b = BCell.clone(self)
            b.mutate_FR()
            new_clones.append(b)

        # return the result
        if (self.nb>0): new_clones.append(self)
        return new_clones


###### Main functions ######


def usage():
    print("")


def main(verbose=False):
    """ Simulate the affinity maturation process in a single germinal center (GC) and save the results to a CSV file. """
    
    # Run multiple trials and save all data to file
        
    start    = timer()
    
    fseed = open('seed.csv','w')
    fendmut = open('output-endmut.csv', 'w')
    fend  = open('output-end.csv', 'w')
    ftot  = open('output-total.csv',  'w')
    fbig  = open('output-largest-clone.csv', 'w')
    fsurv = open('output-surv.csv','w')
    
    fend.write('trial,exit_cycle,number,generation,CDR_mutations,E,Ec,breadth,res0,res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12,res13,res14,res15,res16,res17,res18,res19,res20,res21,res22,res23,res24,res25,res26,res27,res28,res29,res30,res31,res32,res33,res34,res35,res36,res37,res38,res39,res40,res41,res42,res43,res44,res45\n')
    fendmut.write('trial,exit_cycle,number,generation,CDR_mutations,E,Ec,breadth,mut,res0,res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12,res13,res14,res15,res16,res17,res18,res19,res20,res21,res22,res23,res24,res25,res26,res27,res28,res29,res30,res31,res32,res33,res34,res35,res36,res37,res38,res39,res40,res41,res42,res43,res44,res45\n')
    ftot.write('trial,cycle,number recycled,number exit,mean E,mean Ec,mean breadth,mean nb CDR mut\n')
    fbig.write('trial,cycle,update,generation,CDR_mutations,E,delta_res,mut_res_index,res0,res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12,res13,res14,res15,res16,res17,res18,res19,res20,res21,res22,res23,res24,res25,res26,res27,res28,res29,res30,res31,res32,res33,res34,res35,res36,res37,res38,res39,res40,res41,res42,res43,res44,res45\n')
    fsurv.write('trial,cycle,survival rate\n')
    fseed.write('trial,exit_cycle,number,generation,CDR_mutations,E,Ec,breadth,mut,res0,res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12,res13,res14,res15,res16,res17,res18,res19,res20,res21,res22,res23,res24,res25,res26,res27,res28,res29,res30,res31,res32,res33,res34,res35,res36,res37,res38,res39,res40,res41,res42,res43,res44,res45\n')


    # Events of a trial
    for t in range(nb_trial):
    
        print_update(t, nb_trial)   # status check

        # INITIALIZATION - DEFINE DATA STRUCTURES

        recycled_cells   = []
        exit_cells       = [] # cells at the end of the simulation
        memory_cells     = [] # exit cells from previous cycles
        nb_recycled      = []
        nb_exit          = []
        memory_founders  = []
        GC_survival_rate = [] # nb of B cells after selection / nb of B cells after dark zone 

        # CYCLES 1 + 2 - CREATE FOUNDERS AND REPLICATE WITHOUT MUTATION
        
        nb_founders = nb_GC_founders #3   # number of founder B cells for a GC
        id_seeding_cells = np.random.choice(len(seedingCells), nb_founders, replace = False)
        print(id_seeding_cells)        
        B_cells = [BCell(res = seedingCells[id_seeding_cells[i]]) for i in range(nb_founders)]
                
        # Update data
        #cycle 0
        nb_recycled.append(nb_founders)                     # all founders are recycled
        nb_exit.append(0)                                   # no founders exit the GC
        recycled_cells.append([deepcopy(b) for b in B_cells]) # add all cells of all 3 clones       
        
        #cycle 1
        nb_recycled.append(np.sum([b.nb for b in B_cells])) # all founders replicate and are recycled
        recycled_cells.append([deepcopy(b) for b in B_cells]) # add all cells of all 3 clones
        nb_exit.append(0)                                   # no founders exit
        
        # AFFINITY MATURATION
        
        GC_size_max  = nb_recycled[-1]  # maximum number of cells in the GC (= initial population size)
        cycle_number = 2
        if flag == 1:
            nb_cycle_max = 199
        else:
            nb_cycle_max = len(dicAgs)+ cycle_number -1     # maximum number of GC cycles
        print(nb_cycle_max)
        while (cycle_number < nb_cycle_max): 
        #for cycle_number in range(2, nb_cycle_max):      
             
            cycleAntigens = np.array(dicAgs[cycle_number])
            nb_Ag = find_nb_Ag(cycleAntigens)           
            cycleconc = dicconc[cycle_number]
            cycledur  = dicGCDur[cycle_number]

            if cycle_number < cycledur:
                # keep same GC
                B_cells, out_cells, GC_surv_fraction = run_GC_cycle(B_cells, cycleAntigens, cycleconc, nb_Ag, cycle_number)
                GC_survival_rate.append(GC_surv_fraction)
            elif cycle_number == cycledur:
                # start new GC
                print('starting new GC at cycle number %d' % (cycle_number))
                memory_founders = pick_memCells_for_new_GC(memory_cells, nb_GC_founders) 
                for b in memory_founders:
                    fseed.write('%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf' % (t, b.cycle_number, b.nb, b.generation, b.nb_CDR_mut, b.E, b.Ec, b.breadth,b.res[0], b.res[1], b.res[2], b.res[3], b.res[4], b.res[5], b.res[6], b.res[7], b.res[8], b.res[9], b.res[10], b.res[11], b.res[12], b.res[13], b.res[14], b.res[15], b.res[16], b.res[17], b.res[18], b.res[19],b.res[20], b.res[21], b.res[22], b.res[23], b.res[24], b.res[25], b.res[26], b.res[27], b.res[28], b.res[29],b.res[30], b.res[31], b.res[32], b.res[33], b.res[34], b.res[35], b.res[36], b.res[37], b.res[38], b.res[39],b.res[40], b.res[41], b.res[42], b.res[43], b.res[44], b.res[45]))
                    fseed.write('\n')
                B_cells, out_cells, GC_surv_fraction = run_GC_cycle(memory_founders, cycleAntigens, cycleconc, nb_Ag, cycle_number)
                GC_survival_rate.append(GC_surv_fraction)
            else: 
                print('error in starting a GC')
                print(cycle_number)                
            
            GC_size = np.sum([b.nb for b in B_cells])       # total number of cells in the GC
            
            if (cycle_number == nb_cycle_max-1) or (GC_size>GC_size_max): # at the end, all B cells exit the GC
                out_cells += B_cells
                nb_exit.append(np.sum([b.nb for b in out_cells]))
            else:
                memory_cells += out_cells
                nb_exit.append(np.sum([b.nb for b in out_cells]))
                out_cells = []
            
            recycled_cells.append([deepcopy(b) for b in B_cells])
            exit_cells.append(out_cells)
            nb_recycled.append(GC_size)
           
            if (nb_recycled[-1] == 0):break
            elif (GC_size>GC_size_max): cycle_number = cycledur
            else: cycle_number += 1
            
            fseed.flush()
              
        for i in range(len(exit_cells)):
            for b in exit_cells[i]:
                if b.nb>50:
                    fend.write('%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf' % (t, b.cycle_number, b.nb, b.generation, b.nb_CDR_mut, b.E, b.Ec, b.breadth,b.res[0], b.res[1], b.res[2], b.res[3], b.res[4], b.res[5], b.res[6], b.res[7], b.res[8], b.res[9], b.res[10], b.res[11], b.res[12], b.res[13], b.res[14], b.res[15], b.res[16], b.res[17], b.res[18], b.res[19],b.res[20], b.res[21], b.res[22], b.res[23], b.res[24], b.res[25], b.res[26], b.res[27], b.res[28], b.res[29],b.res[30], b.res[31], b.res[32], b.res[33], b.res[34], b.res[35], b.res[36], b.res[37], b.res[38], b.res[39],b.res[40], b.res[41], b.res[42], b.res[43], b.res[44], b.res[45]))
                    fend.write('\n')
                    fendmut.write('%d,%d,%d,%d,%d,%lf,%lf,%lf,%s,%s,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf' % (t, b.cycle_number, b.nb, b.generation, b.nb_CDR_mut, b.E, b.Ec, b.breadth,b.history['mut_res_id'],b.history['delta_e'],b.res[0], b.res[1], b.res[2], b.res[3], b.res[4], b.res[5], b.res[6], b.res[7], b.res[8], b.res[9], b.res[10], b.res[11], b.res[12], b.res[13], b.res[14], b.res[15], b.res[16], b.res[17], b.res[18], b.res[19],b.res[20], b.res[21], b.res[22], b.res[23], b.res[24], b.res[25], b.res[26], b.res[27], b.res[28], b.res[29],b.res[30], b.res[31], b.res[32], b.res[33], b.res[34], b.res[35], b.res[36], b.res[37], b.res[38], b.res[39],b.res[40], b.res[41], b.res[42], b.res[43], b.res[44], b.res[45]))
                    fendmut.write('\n')

        fend.flush()
        fendmut.flush()        

        for i in range(len(GC_survival_rate)):
            fsurv.write('%d,%d,%lf' % (t, i+2,GC_survival_rate[i]))
            fsurv.write('\n')
        fsurv.flush()

        for i in range(len(recycled_cells)):    
            meanE = 0
            meanEc = 0
            meanBreadth = 0
            meanCDRMutations = 0
            count_clones = 0
            if nb_recycled[i] > 0:
                for b in recycled_cells[i]:
                    count_clones += 1
                    meanE += b.E
                    meanEc += b.Ec
                    meanBreadth += b.breadth
                    meanCDRMutations += b.nb_CDR_mut          
                meanE /= count_clones
                meanEc /= count_clones
                meanBreadth /= count_clones
                meanCDRMutations /= count_clones
                cycle = recycled_cells[i][0].cycle_number
                
            ftot.write('%d,%d,%d,%d,%lf,%lf,%lf,%lf\n' % (t, cycle, nb_recycled[i],nb_exit[i], meanE, meanEc, meanBreadth, meanCDRMutations))
        ftot.flush()
            
    # End and output total time

        if len(exit_cells[-1])>0:
            for j in range(len(exit_cells[-1])):
                b   = exit_cells[-1][j]
                if b.nb>50:
                    with open('history_largest_clone.csv', 'w') as h:
                        w = csv.writer(h)
                        w.writerows(b.history.items())
             
                    for i in range(len(b.history['E'])):
                        fbig.write('%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n' % (t, b.history['cycle_number'][i], i, b.history['generation'][i], b.history['nb_CDR_mut'][i], b.history['E'][i], b.history['delta_res'][i], b.history['mut_res_id'][i], b.history['res'][i][0], b.history['res'][i][1], b.history['res'][i][2], b.history['res'][i][3], b.history['res'][i][4], b.history['res'][i][5], b.history['res'][i][6], b.history['res'][i][7], b.history['res'][i][8], b.history['res'][i][9], b.history['res'][i][10], b.history['res'][i][11], b.history['res'][i][12], b.history['res'][i][13], b.history['res'][i][14], b.history['res'][i][15], b.history['res'][i][16], b.history['res'][i][17], b.history['res'][i][18], b.history['res'][i][19],b.history['res'][i][20], b.history['res'][i][21], b.history['res'][i][22], b.history['res'][i][23], b.history['res'][i][24], b.history['res'][i][25], b.history['res'][i][26], b.history['res'][i][27], b.history['res'][i][28], b.history['res'][i][29],b.history['res'][i][30], b.history['res'][i][31], b.history['res'][i][32], b.history['res'][i][33], b.history['res'][i][34], b.history['res'][i][35], b.history['res'][i][36], b.history['res'][i][37], b.history['res'][i][38], b.history['res'][i][39],b.history['res'][i][40], b.history['res'][i][41], b.history['res'][i][42], b.history['res'][i][43], b.history['res'][i][44], b.history['res'][i][45]))
                    fbig.write('%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n' % (t, b.cycle_number, len(b.history['E']), b.generation, b.nb_CDR_mut, b.E, b.delta_res, b.mut_res_id, b.res[0], b.res[1], b.res[2], b.res[3], b.res[4], b.res[5], b.res[6], b.res[7], b.res[8], b.res[9], b.res[10], b.res[11], b.res[12], b.res[13], b.res[14], b.res[15], b.res[16], b.res[17], b.res[18], b.res[19],b.res[20], b.res[21], b.res[22], b.res[23], b.res[24], b.res[25], b.res[26], b.res[27], b.res[28], b.res[29],b.res[30], b.res[31], b.res[32], b.res[33], b.res[34], b.res[35], b.res[36], b.res[37], b.res[38], b.res[39],b.res[40], b.res[41], b.res[42], b.res[43], b.res[44], b.res[45]))
            fbig.flush()

    fend.close()
    fendmut.close()
    ftot.close()
    fbig.close()
    fsurv.close()
    fseed.close()    

    end = timer()
    print('\nTotal time: %lfs, average per cycle %lfs' % ((end - start),(end - start)/float(nb_trial)))

def find_nb_Ag(antigens):
    if np.shape(antigens)[0]==length: nb_Ag=1
    else:                             nb_Ag=np.shape(antigens)[0]
    return nb_Ag
       
def print_update(current, end, bar_length=20):
    """ Print an update of the simulation status. h/t Aravind Voggu on StackOverflow. """
    
    percent = float(current) / end
    dash    = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
    space   = ''.join([' ' for k in range(bar_length - len(dash))])

    sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
    sys.stdout.flush()

def updating_antigens(B_cells, cycleAntigens):
    """ The antigens for all B cells are updated with the beginning of a new cycle. """
    for b in B_cells:
        b.update_antigens(cycleAntigens)    
    return B_cells
    
def run_dark_zone(B_cells, cycle_number, nb_rounds = 2):
    """ B cells proliferate and undergo SHM in the dark zone. """
    
    for i in range(nb_rounds):
        new_cells = []
        for b in B_cells:
            b.divide(cycle_number)
            new_cells += b.shm(cycle_number)
        B_cells = new_cells
    return B_cells

def run_binding_selection(B_cells, cycleconc, nb_Ag):
    """ Select B cells for binding to antigen. """
    
    new_cells=[]
    for b in B_cells:

        b.last_bound = np.random.multinomial(b.nb, pvals = [1./float(nb_Ag)] * nb_Ag)
        
        for i in range(nb_Ag):
            
            # compute binding energy and chance of death ( = 1 - chance of survival )
            Ag_bound      = np.exp(energy_scale * (b.energy(b.antigens[i])-activation_energy))
            factor        = cycleconc * Ag_bound
            langmuir_conj = 1. / (1. + factor)
            
            # remove dead cells and update binding details
            n_die            = np.random.binomial(b.last_bound[i], langmuir_conj)
            b.nb            -= n_die
            b.last_bound[i] -= n_die
        if b.nb>0:new_cells.append(b)
    return new_cells

def run_help_selection(B_cells, nb_Ag):
    """ Select B cells to receive T cell help. """
    #nb_Ag = B_cells[0].nb_Ag
    
    # get binding energies
    binding_energy     = [[b.energy(b.antigens[i]) for i in range(nb_Ag)] for b in B_cells]
    binding_energy_tot = []
    for i in range(len(B_cells)):
        for j in range(nb_Ag): binding_energy_tot += [binding_energy[i][j]] * B_cells[i].last_bound[j]
    
    # cells in the top (help_cutoff) fraction of binders survive
    if len(binding_energy_tot)>0:
        cut_idx       = np.max([0, int(np.floor(help_cutoff * len(binding_energy_tot)))-1])
        energy_cutoff = np.array(binding_energy_tot)[np.argsort(binding_energy_tot)][::-1][cut_idx]
        n_die_tie     = len(binding_energy_tot) - cut_idx - np.sum(binding_energy_tot < energy_cutoff)

        # kill all B cells below threshold
        for i in np.random.permutation(len(B_cells)):
            for j in np.random.permutation(nb_Ag):
                energy = binding_energy[i][j]
                if energy < energy_cutoff:
                    B_cells[i].nb            -= B_cells[i].last_bound[j]
                    B_cells[i].last_bound[j]  = 0
                elif (energy == energy_cutoff) and (n_die_tie > 0):
                    if B_cells[i].last_bound[j] < n_die_tie:
                        B_cells[i].nb            -= B_cells[i].last_bound[j]
                        n_die_tie                -= B_cells[i].last_bound[j]
                        B_cells[i].last_bound[j]  = 0
                    else:
                        B_cells[i].nb            -= n_die_tie
                        B_cells[i].last_bound[j] -= n_die_tie
                        n_die_tie                 = 0
    cells_surv = np.sum([b.nb for b in B_cells])   
    return B_cells, cells_surv
    

def run_recycle(B_cells):
    """ Randomly select B cells to be recycled back into the GC or to exit. """

    new_cells  = []                                 # cells that will remain in the GC
    exit_cells = []                                 # cells that will exit the GC
    n_tot      = np.sum([b.nb for b in B_cells])    # total number of cells currently in the GC
    n_exit     = int(np.floor(p_exit * n_tot))      # number of cells that will exit the GC
    b_exit     = np.array([])                       # index of cells that will exit the GC

    if (n_tot > 0) and (n_exit > 0):
        b_exit = np.random.choice(n_tot, n_exit, replace=False)

    idx = 0
    for b in B_cells:
    
        # find which cells exit the GC
        n_exit  = np.sum((idx <= b_exit) * (b_exit < idx + b.nb))
        idx    += b.nb
        b.nb   -= n_exit
        
        # add remainder to recycled cells
        if (b.nb > 0):
            new_cells.append(b)
    
        # record exit cells
        if (n_exit > 0):
            exit_cells.append(deepcopy(b))
            exit_cells[-1].nb = n_exit

    return new_cells, exit_cells

def pick_memCells_for_new_GC(memory_cells, nb_GC_founders):
    n_mem_cells = len(memory_cells)
    id_new_founders = np.random.choice(n_mem_cells, nb_GC_founders, replace=False)
    new_founders = [memory_cells[id_new_founders[i]] for i in range(nb_GC_founders)]
    return new_founders

def run_breadth_calculation(panel_energies, threshold, panelSize):
    average  = np.mean(panel_energies)
    variance = np.var(panel_energies)
    breadth  = float(sum(x > threshold for x in panel_energies))/panelSize 
    return average, variance, breadth

def run_GC_cycle(B_cells, cycleAntigens, cycleconc, nb_Ag, cycle_number):
    """ Run one cycle of the GC reaction. """
    B_cells = updating_antigens(B_cells, cycleAntigens)         # UPDATE antigens
    B_cells = run_dark_zone(B_cells, cycle_number)              # DARK  ZONE - two rounds of division + SHM + updates cycle_number
    total_cells = np.sum([b.nb for b in B_cells])
    
    if total_cells == 0: 
       print('GC extinct at cycle ', cycle_number)
       exit_cells = [] 
       GC_surv_fraction = 0
    else: 
        B_cells = run_binding_selection(B_cells, cycleconc, nb_Ag)  # LIGHT ZONE - selection for binding to Ag
        B_cells, cells_surv = run_help_selection(B_cells, nb_Ag)    # LIGHT ZONE - selection to receive T cell help
        GC_surv_fraction = float(float(cells_surv)/float(total_cells))
        B_cells, exit_cells = run_recycle(B_cells)
    
    return B_cells, exit_cells, GC_surv_fraction               # RECYCLE    - randomly pick exiting cells from the surviving B cells



if __name__ == '__main__': main()



