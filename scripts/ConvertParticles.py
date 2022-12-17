"""
Adapted from BRIO lab/Shoulder Keypiont
Edited by Sasank Desaraju
"""

from copyreg import remove_extension
import os
import glob
import shutil
import re

HOME_DIR = '/media/sasank/LinuxStorage/Dropbox (UFL)/Canine Kinematics Data/'
#HOME_DIR = 'C:/Users/sasan/Dropbox (UFL)/Canine Kinematics Data/'
STUDY = 'TPLO_Ten_Dogs/'
KNEE_PATH = HOME_DIR + STUDY
BONE = 'tib'        # 'fem' or 'tib'
if BONE == 'fem':
    #PARTICLE_PATH = HOME_DIR + 'Shapeworks/Projects/Femur_Keypoints_particles/'
    PARTICLE_PATH = HOME_DIR + 'Shapeworks/Projects/old_Femur_Keypoints_particles/'
elif BONE == 'tib':
    PARTICLE_PATH = HOME_DIR + 'Shapeworks/Projects/Tibia_16KPs_particles/'



"""
This function is used to convert the .particles files to .kp files
to use in StudyToGrid.
"""

def get_mvt_dirs(HOME_DIR,STUDY):
    mvt_dirs = []
    pat_nums = []
    studies = []
    patients = []
    sessions = []
    study = STUDY
    study_dir = HOME_DIR + study
    study_org_dir = study_dir
    # Lima folders have a bit of a different structure
    # Looping through patients within each study using list comprehension
    # We want to make sure that we are only grabbing directories 
    for pat_id in [x for x in sorted(os.listdir(study_org_dir)) if os.path.isdir(study_org_dir + "/" + x)]:
        #pat_num = re.search('\d{0,2}', pat_id)
        pat_num = re.search('(?:(?<=Patient_))\d{0,2}', pat_id).group(0)
        
        pat_dir = study_org_dir + pat_id
        # looping through each session using list comprehension
        for sess_id in [x for x in os.listdir(pat_dir) if os.path.isdir(pat_dir + "/" + x)]:
            sess_dir = pat_dir + "/" + sess_id
            for mvt_id in [x for x in os.listdir(sess_dir) if (os.path.isdir(sess_dir + "/" + x))]:
                mvt_dir = sess_dir + "/" + mvt_id + "/"
                mvt_dirs.append(mvt_dir)
                pat_nums.append(pat_num)
    
    return mvt_dirs

# shoulderPath = input("Please input the direct path to the folder containing both shoulder datasets: ")
kneePath = KNEE_PATH
# Swapping direction of backslashes to forward slashes
# shoulderPath = shoulderPath.replace('\\', '/')    # Might be for Windows

# parts = input("Please input the direct path to the folder containing particle folders: ")
parts = PARTICLE_PATH
# parts = parts.replace('\\', '/')      # Might be for Windows

# bone = input("Enter 1 for scapulae, 2 for humerus, or 3 for clavicle: ")
bone = BONE    # 0 for femur
if bone == '1':
    parts=parts+"/all-scaps_particles/"
if bone == '2':
    parts=parts+"/all-hums_particles/"
if bone == '3':
    parts=parts+"/all-clav_particles/"
    
# Generate a list of paths to each group of .stl files
study_list = ["TPLO_Ten_Dogs"]      # Idk if this is getting used
dirs = get_mvt_dirs(HOME_DIR, STUDY)
dirs = sorted(dirs)
print(dirs)
patientCount = len(dirs)
print("Number of movements: ", patientCount)

for mvt_idx, mvt_dir in enumerate(dirs):
    # Change the cwd to the path to each group of .stls
    os.chdir(dirs[mvt_idx])
    print("Movement: " + dirs[mvt_idx])
    if bone == 'fem':
        
        # Find what patient we are in
        pat_id = re.search('(?:(?<=Patient_))\d{0,2}', mvt_dir).group(0)

        # Find what session we are in
        sess_id = re.search('(?:(?<=Session_))\d{0,2}', mvt_dir).group(0)

        TEST_DIR = HOME_DIR + 'Shapeworks/scripts/test/'

        # If session is Preop or Postop, then use the Preop particle file
        if sess_id == '1' or sess_id == '3':
            # Find the right Preop particle file
            part_file_path = glob.glob(PARTICLE_PATH + 'P' + pat_id + '[L,R]' + "_Pre_femur_groomed_local.particles")[0]
        elif sess_id == '2':
            # Skip for P6L_Contra_femur bc that one's Groom doesn't work in ShapeWorks
            # TODO: Figure out why and fix it
            if pat_id == '6' and sess_id == '2':
                continue
            # Find the right Contra particle file
            part_file_path = glob.glob(PARTICLE_PATH + 'P' + pat_id + '[L,R]' + "_Contra_femur_groomed_local.particles")[0]
        else:
            raise ValueError('Session ID is not 1, 2, or 3')

        # Snip just the file name from the full file name path
        #part_file_name = part_file_path.split('/')[-1]

        # Create kp file in mvt_dir with proper top and bottom lines (BEGIN_KP and END_KP) like StudyToGrid wants
        part_file = open(part_file_path, 'r')
        kp_file = open(mvt_dir + 'fem.kp', 'w')
        part_lines = part_file.readlines()
        kp_file.write('BEGIN_KP\n')
        for idx, line in enumerate(part_lines):
            kp_file.write(str(idx) + ':' + line)
        kp_file.write('END_KP')
        kp_file.close()
        part_file.close()

        # Read in the particle file from part_file_path

        # Create the new file at mvt_dir + 'fem.kp'

        # Add BEGIN_KP at the top of the new file

        # Copy the particle file contents into the new file

        # Add END_KP at the bottom of the new file




        # Copy the Preop particle file to the current directory
        #shutil.copy(part_file_path, mvt_dir + part_file_name)
        #print('Copied ' + part_file_name + ' to ' + mvt_dir + part_file_name)
        #shutil.copy(part_file_path, TEST_DIR + part_file_name)

        # Rename the particle file to fem.kp
        #os.rename(mvt_dir + part_file_name, mvt_dir + 'fem.kp')
        #os.rename(TEST_DIR + part_file_name, TEST_DIR + str(mvt_idx) + 'fem.kp')


    if bone == 'tib':
        # Find what patient we are in
        pat_id = re.search('(?:(?<=Patient_))\d{0,2}', mvt_dir).group(0)

        # Find what session we are in
        sess_id = re.search('(?:(?<=Session_))\d{0,2}', mvt_dir).group(0)

        TEST_DIR = HOME_DIR + 'Shapeworks/scripts/test/'

        if sess_id == '1' or sess_id == '3':
            # Find the right Preop particle file
            part_file_path = glob.glob(PARTICLE_PATH + 'P' + pat_id + '[L,R]' + "_Pre_tibia_groomed_local.particles")[0]
        elif sess_id == '2':
            # Find the right Contra particle file
            part_file_path = glob.glob(PARTICLE_PATH + 'P' + pat_id + '[L,R]' + "_Contra_tibia_groomed_local.particles")[0]
        else:
            raise ValueError('Session ID is not 1, 2, or 3')

        # Snip just the file name from the full file name path
        #part_file_name = part_file_path.split('/')[-1]

        # Create kp file in mvt_dir with proper top and bottom lines (BEGIN_KP and END_KP) like StudyToGrid wants
        part_file = open(part_file_path, 'r')
        kp_file = open(mvt_dir + 'tib.kp', 'w')
        part_lines = part_file.readlines()
        kp_file.write('BEGIN_KP\n')
        for idx, line in enumerate(part_lines):
            kp_file.write(str(idx) + ':' + line)
        kp_file.write('END_KP')
        kp_file.close()
        part_file.close()

    

    if bone == '1':
        for file in glob.glob('*sca.stl'):
            #print(file)
            file = os.path.splitext(file)[0]
            #print(file)
            file = file+"_groomed_local.particles"
            #print(file)
            # print(parts+file)
            for file2 in glob.glob(parts+file):
                # print(file2)
                shutil.copy(file2,dirs[mvt_idx])
                os.replace(dirs[mvt_idx] +"/"+ file, "sca.kp")
    elif bone == '2':
        for file in glob.glob('*hum.stl'):
            file = os.path.splitext(file)[0]
            #print(file)
            file = file+"_groomed_local.particles"
            #print(file)
            # print(parts+file)
            for file2 in glob.glob(parts+file):
                # print(file2)
                shutil.copy(file2,dirs[mvt_idx])
                os.replace(dirs[mvt_idx] +"/"+ file, "hum.kp")
    elif bone == '3':
        for file in glob.glob('*cla.stl'):
            file = os.path.splitext(file)[0]
            #print(file)
            file = file+"_groomed_local.particles"
            #print(file)
            # print(parts+file)
            for file2 in glob.glob(parts+file):
                # print(file2)
                shutil.copy(file2,dirs[mvt_idx])
                os.replace(dirs[mvt_idx] +"/"+ file, "cla.kp")
    
print("Particles converted successfully.\n.kp and .jts files are located in each movement directory, for each side (L and R) that exists per patient.")