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
import sys
from progress.bar import IncrementalBar

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
os.makedirs(os.path.join(newpath,"hist"))


# & SYSTEM ####################################################################################################################################
pdb = app.PDBFile(pdbname)

# ? if its PFOA or proteins of superset of PFOA, it might have some missing hydrogens
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('amber14-all.xml')
modeller.addHydrogens(forcefield)
pdb = modeller

# * system and simulation definition
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds, hydrogenMass=1.5*unit.amu)
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(app.DCDReporter(os.path.join(newpath, 'hist', 'smd_traj.dcd'), 10000))
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
simulation.step(1000)
#&##################################################################################################################################

# ~ PARAMS ######################################################################################################

# * CV := distance betw CAs of two end residues
L_i = 1.3 # ? start length
L_f = 3.3 # ? end length
index1 = 8
index2 = 98
cv = mm.CustomBondForce('r')
cv.addBond(index1, index2)
num_win = 2

r0 = L_i*unit.nanometers #start value
fc_pull = 1000.0*unit.kilojoules_per_mole/unit.nanometers**2 #force constnat
v_pulling = 0.02*unit.nanometers/unit.picosecond #pulling velocity
dt = simulation.integrator.getStepSize() # simulation time step



total_steps = 30000 # ? total steps
increment_steps = 10 # ? step size
wTotal = 100000
wdelta = 1000 # ? record steps


# * Harmonic force definiton
pullingForce = mm.CustomCVForce('0.5 * fc_pull * (cv-r0)^2')
pullingForce.addGlobalParameter('fc_pull', fc_pull)
pullingForce.addGlobalParameter('r0', r0)
pullingForce.addCollectiveVariable("cv", cv)
system.addForce(pullingForce)
simulation.context.reinitialize(preserveState=True)

# * Window definition
windows = np.linspace(L_i, L_f, num_win)
window_coords = []
window_index = 0


##~############################################################################################################################

#!########## SIMULATION ##################################################################################################################

# * SMD pulling loop
# Define the progress bar
progress_bar = IncrementalBar('Pull Loop', max=total_steps//increment_steps)

# Redirect stdout to a file or another stream
original_stdout = sys.stdout
with open('output.log', 'w') as f:
    sys.stdout = f

    # SMD pulling loop
    for i in range(total_steps//increment_steps):
        # Your simulation steps here
        simulation.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
        r0 += v_pulling * dt * increment_steps
        simulation.context.setParameter('r0', r0)
        if (window_index < len(windows) and current_cv_value >= windows[window_index]):
            window_coords.append(simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions())
            window_index += 1
        progress_bar.next()

    # Finish the progress bar
    progress_bar.finish()

# Reset stdout to its original value
sys.stdout = original_stdout

#simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))

# * Save Windows
for i, coords in enumerate(window_coords):
   # Save window structures to 'windows' directory
   window_outfile = open(os.path.join(newpath,"windows", f'window_{i}.pdb'), 'w')
   app.PDBFile.writeFile(simulation.topology, coords, window_outfile)
   window_outfile.close()

#!##################################################################################################################################

##^## RUN WINDOWS AND REUSE SIMULATION -> CV TIME SERIES FILES ########################################################################################################################################

def run_window(window_index):
    pdb = app.PDBFile(os.path.join(newpath, "windows", f'window_{window_index}.pdb'))  # Load from 'windows' directory
    simulation.context.setPositions(pdb.positions)
    r0 = windows[window_index]
    simulation.context.setParameter('r0', r0)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    simulation.step(wdelta)
    total_steps = wTotal
    record_steps = wdelta
    cv_values = []

    # Define the progress bar
    progress_bar = IncrementalBar(f'Window {window_index}', max=total_steps//record_steps)

    # SMD pulling loop
    for i in range(total_steps//record_steps):
        simulation.step(record_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
        cv_values.append([i, current_cv_value[0]])
        
        # Update the progress bar
        progress_bar.next()

    # Finish the progress bar
    progress_bar.finish()

    # Save CV values to file
    np.savetxt(os.path.join(newpath, "cv", f'cv_{window_index}.txt'), np.array(cv_values))  # Save to 'cv' directory


for n in range(num_win):
   run_window(n)

#^#######################################################################################################################################

# & histograms write ##########################################################################################################################################
metafilelines = []
for i in range(len(windows)):
    cvpath = os.path.join(newpath,"cv",f'cv_{i}.txt')
    data = np.loadtxt(f'{cvpath}')
    plt.hist(data[:,1])
    metafileline = f'cv_{i}.txt {windows[i]}' + str(wdelta) + '\n' # ? do i do just 'cv_{i}.txt {windows[i]}' or '{cvpath}' ?
    metafilelines.append(metafileline)

plt.xlabel("r (nm)")
plt.ylabel("count")

with open(os.path.join(newpath,"hist","metafile.txt"), "w") as f:
    f.writelines(metafilelines)
#&#######################################################################################################################################################################################

# ~ WHAM on metafile.txt ################################################################################################################################################################

# ! USE POPEN TO EXECUTE child program ---->>>>> /Users/edwardkim/Downloads/wham 1.3 3.3 50 1e-6 300 0 metafile.txt pmf.txt > wham_log.txt

args = ["/Users/edwardkim/Desktop/scoresolver/wham/wham/wham", str(L_i), str(L_f), '50', '1e-6', '300', '0', os.path.join(newpath,"hist", "metafile.txt"), os.path.join(newpath,"hist", "pmf.txt")]


os.chmod("/Users/edwardkim/Desktop/scoresolver/wham/wham/wham", 0o755)
# Execute the command and store the Popen object
process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for the process to terminate and capture stdout and stderr
stdout_data, stderr_data = process.communicate()

# Write stdout to wham_logs.txt
with open(os.path.join(newpath,"hist","wham_logs.txt"), "w") as output_file:
    output_file.write(stdout_data.decode())  # Decode stdout_data from bytes to string

# Check if there are any errors
if stderr_data:
    print("Error occurred:", stderr_data.decode())  # Decode stderr_data from bytes to string

# ~ ###############################################################################################################################################################

# & plot PMF

open(os.path.join(newpath,"hist","pmf.txt"), "w")
# Load the PMF data
pmf = np.loadtxt(os.path.join(newpath, "hist", "pmf.txt"))

# Check the shape of the array
print("Shape of pmf:", pmf.shape)

# If pmf is one-dimensional, plot it directly
if pmf.ndim == 1:
    plt.plot(pmf)
    plt.xlabel("Index")
    plt.ylabel("PMF (kJ/mol)")
    plt.title("Potential of Mean Force (PMF)")
    plt.show()
# If pmf has two dimensions, plot the first column against the second
elif pmf.ndim == 2:
    plt.plot(pmf[:, 0], pmf[:, 1])
    plt.xlabel("r (nm)")
    plt.ylabel("PMF (kJ/mol)")
    plt.title("Potential of Mean Force (PMF)")
    plt.show()
else:
    print("Unexpected shape of pmf array:", pmf.shape)
