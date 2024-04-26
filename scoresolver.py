import MDAnalysis as mda
import openmm as mm
from openmm import unit
import numpy as np
from sys import stdout
import matplotlib.pyplot as plt
import openmm.app as app
from openmm.app import Modeller
import subprocess
import os
import re

# & make new test folder
masterpath = "/Users/edwardkim/Desktop/scoresolver"
pdbname = "deca.pdb"

dir_list = os.listdir(masterpath)

def increment_test_string(test_string):
    # Regular expression pattern to match "test_" followed by digits
    pattern = re.compile(r'test_(\d+)')
    
    match = pattern.match(test_string)
    if match:
        index = int(match.group(1))
        new_index = index + 1
        new_string = f"test_{new_index}"
        return new_string
    else:
        return None

def nextFolder(): 
    max_index = -1
    max_folder = None
    pattern = re.compile(r'test_(\d+)')

    for folder in dir_list:
        match = pattern.match(folder)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                max_folder = folder

    if max_folder == None:
        return "test_0"
    return increment_test_string(max_folder)

testname = nextFolder()
newpath = os.path.join(masterpath, testname)

if not os.path.exists(newpath):
    os.makedirs(newpath)

os.makedirs(os.path.join(newpath,"cv"))
os.makedirs(os.path.join(newpath,"windows"))


#### SYSTEM ####################################################################################################################################
pdb = app.PDBFile(pdbname)

modeller = Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('amber14-all.xml')
modeller.addHydrogens(forcefield)
pdb = modeller

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds, hydrogenMass=1.5*unit.amu)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.DCDReporter(os.path.join(newpath, 'smd_traj.dcd'), 10000))
simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))


simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
simulation.step(1000)
####################################################################################################################################


################ PARAMS ######################################################################################################


# CV := distance betw CAs of two end residues
index1 = 8
index2 = 98
cv = mm.CustomBondForce('r')
cv.addBond(index1, index2)


r0 = 1.3*unit.nanometers #start value
fc_pull = 1000.0*unit.kilojoules_per_mole/unit.nanometers**2 #force constnat
v_pulling = 0.02*unit.nanometers/unit.picosecond #pulling velocity
dt = simulation.integrator.getStepSize() # simulation time step


# SMD Pull loop
total_steps = 30000 #total steps
increment_steps = 10 #step size


# Window loop
wTotal = 100000
wdelta = 1000


# define a harmonic restraint on the CV
# the location of the restrain will be moved as we run the simulation
# this is constant velocity steered MD
pullingForce = mm.CustomCVForce('0.5 * fc_pull * (cv-r0)^2')
pullingForce.addGlobalParameter('fc_pull', fc_pull)
pullingForce.addGlobalParameter('r0', r0)
pullingForce.addCollectiveVariable("cv", cv)
system.addForce(pullingForce)
simulation.context.reinitialize(preserveState=True)


# define the windows
# during the pulling loop we will save specific configurations corresponding to the windows
L_i = 1.3
L_f = 3.3

windows = np.linspace(L_i, L_f, 24)
window_coords = []
window_index = 0


##############################################################################################################################


############ SIMULATION ##################################################################################################################


# SMD pulling loop
for i in range(total_steps//increment_steps):
   simulation.step(increment_steps)
   current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)


   # increment the location of the CV based on the pulling velocity
   r0 += v_pulling * dt * increment_steps
   simulation.context.setParameter('r0',r0)


   # check if we should save this config as a window starting structure
   if (window_index < len(windows) and current_cv_value >= windows[window_index]):
       window_coords.append(simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions())
       window_index += 1


for i, coords in enumerate(window_coords):
   # Save window structures to 'windows' directory
   window_outfile = open(os.path.join(newpath,"windows", f'window_{i}.pdb'), 'w')
   app.PDBFile.writeFile(simulation.topology, coords, window_outfile)
   window_outfile.close()


#### RUN WINDOWS AND REUSE SIMULATION -> CV TIME SERIES FILES ########################################################################################################################################
def run_window(window_index):
   print('running window', window_index)
   pdb = app.PDBFile(os.path.join(newpath,"windows", f'window_{window_index}.pdb'))  # Load from 'windows' directory
   simulation.context.setPositions(pdb.positions)
   r0 = windows[window_index]
   simulation.context.setParameter('r0', r0)
   simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
   simulation.step(wdelta)
   total_steps = wTotal
   record_steps = wdelta
   cv_values=[]
   for i in range(total_steps//record_steps):
       simulation.step(record_steps)
       current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
       cv_values.append([i, current_cv_value[0]])
   np.savetxt(os.path.join(newpath,"cv", f'cv_values_{window_index}.txt'), np.array(cv_values))  # Save to 'cv' directory
   print('Completed window', window_index)


for n in range(24):
   run_window(n)

# & histograms write
metafilelines = []
for i in range(len(windows)):
    cvpath = os.path.join(newpath,"cv",f'cv_{i}.txt')
    data = np.loadtxt(f'{cvpath}')
    plt.hist(data[:,1])
    metafileline = f'cv_{i}.txt {windows[i]}' + wdelta + '\n' # ? do i do just 'cv_{i}.txt {windows[i]}' or '{cvpath}' ?
    metafilelines.append(metafileline)

plt.xlabel("r (nm)")
plt.ylabel("count")

with open(os.path.join(newpath,"hist","metafile.txt"), "w") as f:
    f.writelines(metafilelines)

# ~ WHAM on metafile.txt

# ! USE POPEN TO EXECUTE child program ---->>>>> /Users/edwardkim/Downloads/wham 1.3 3.3 50 1e-6 300 0 metafile.txt pmf.txt > wham_log.txt

args = ["/Users/edwardkim/Downloads/wham", L_i, L_f, 50, 1e-6, 300, 0, os.path.join(newpath,"hist", "metafile.txt"), os.path.join(newpath,"hist", "pmf.txt"), ]

subprocess.Popen(args)
open(os.path.join(newpath,"hist","pmf.txt"), "w")
with open(os.path.join(newpath,"hist","wham_logs.txt"), "w") as output_file:
    # Execute the command and redirect the output to the file
    subprocess.Popen(args, stdout=output_file)

# & plot PMF

'''
pmf = np.loadtxt("pmf.txt")
plt.plot(pmf[:,0], pmf[:,1])
plt.xlabel("r (nm)")
plt.ylabel("PMF (kJ/mol)")
plt.show()
'''
