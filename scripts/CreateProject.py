'''
Simple Project creation tool to speed up importing of many .stl files into ShapeWorks
Selects an amount of patients to import selected bones into ShapeWorks.
Assigns groups to these based on their orientation (Left or Right).
'''
from ctypes import sizeof
from re import L
from matplotlib.cm import ScalarMappable
import xlsxwriter
import os
import random
import glob

def get_mvt_dirs(HOME_DIR,study_list):
    cwd = os.getcwd()
    mvt_dirs = []
    pat_nums = []
    studies = []
    patients = []
    sessions = []
    for study in study_list:
        study_dir = HOME_DIR +  "/" + study
        # Lima folders have a bit of a different structure
        if study != "Lima":
            study_org_dir =  study_dir + "_Organized"
        else:
            study_org_dir =  study_dir + "/" + study +  "_Organized_Updated"
        # Looping through patients within each study using list comprehension
        # We want to make sure that we are only grabbing directories 
        for pat_id in [x for x in os.listdir(study_org_dir) if os.path.isdir(study_org_dir + "/" + x)]:
            pat_num = pat_id.removeprefix("Patient_")            
            pat_dir = study_org_dir + "/" + pat_id
            # looping through each session using list comprehension
            for sess_id in [x for x in os.listdir(pat_dir) if os.path.isdir(pat_dir + "/" + x)]:
                sess_dir = pat_dir + "/" + sess_id
                for mvt_id in [x for x in os.listdir(sess_dir) if (os.path.isdir(sess_dir + "/" + x))]:
                    if "2" in mvt_id or "Sup" in mvt_id:   # Avoid importing duplicate sides
                        break
                    mvt_dir = sess_dir + "/" + mvt_id
                    mvt_dirs.append(mvt_dir)
                    pat_nums.append(pat_num)
                break
            # We only need one patient's bone .stl for ShapeWorks, so the looping above is not needed, hence why break appears
    
    return mvt_dirs

# save cwd so that it can be used to create and save the .xlsx later.
cwd = os.getcwd()

# Take input for the name of the file to be saved and the path to shoulder data
filename = input("Please input the name you would like for the dataset file: ")
shoulderPath = input("Please input the direct path to the folder containing both shoulder datasets: ")

# Swapping direction of backslashes to forward slashes
shoulderPath = shoulderPath.replace('\\', '/')

# Initalize the file in the Datasets folder
project = xlsxwriter.Workbook('Projects/'+filename+'.xlsx')
projectSheet = project.add_worksheet()

# Prompt for bone choice
bone = input("Enter 1 for scapulae, 2 for humerus, or 3 for clavicle: ")

# Prompt the user to ask how many patients to import
patientCount = input("How many patients would you like to import? Type \"all\" to import every patient: ")

# Spreadsheet column header
data=["shape_file"]
groupData=["group_side"]

# Create a list of the available studies
study_list = ["Akira", "Keisuke"]

# Generate a list of paths to each group of .stl files
if patientCount != "all":
    patientCount = int(patientCount)
    randDirs = random.sample(get_mvt_dirs(shoulderPath, study_list), patientCount)
else:
    randDirs = get_mvt_dirs(shoulderPath, study_list)
    patientCount = len(randDirs)

# For as many patients as wanted, save the paths to the .stls that are desired
for count in range(0, patientCount-1):
    # Change the cwd to the path to each group of .stls
    os.chdir(randDirs[count])
    if bone == '1':
        for file in glob.glob('*sca.stl'): # Syntax is *sca*.stl in order to account for the renamed "*_reflect.stl" files
            if "Raxes" in file:
                groupData.append("right")
            else:
                groupData.append("left")
            data.append(randDirs[count]+"/"+file)
    if bone == '2':
        for file in glob.glob('*hum.stl'):
            if "Raxes" in file:
                groupData.append("right")
            else:
                groupData.append("left")
            data.append(randDirs[count]+"/"+file)
    if bone == '3':
        for file in glob.glob('*cla.stl'):
            if "Raxes" in file:
                groupData.append("right")
            else:
                groupData.append("left")
            data.append(randDirs[count]+"/"+file)
          
# Change back to the cwd of the file so the project can be saved
os.chdir(cwd)

# Add the data to the .xlsx using xlsxwriter
row = 0
col = 0
for item in (data):
    projectSheet.write(row,col,item)
    row += 1

row = 0
col+=1
for item in (groupData):
    projectSheet.write(row,col,item)
    row += 1

project.close()

print("Project file saved in the Projects folder.")