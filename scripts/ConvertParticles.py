from copyreg import remove_extension
import os
import glob
import shutil

def get_mvt_dirs(HOME_DIR,study_list):
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
            # We only need one patient's bone .stl for ShapeWorks, so the looping above is not needed, hence why break appears twice
    
    return mvt_dirs

study_list = ["Akira", "Keisuke"]

shoulderPath = input("Please input the direct path to the folder containing both shoulder datasets: ")
# Swapping direction of backslashes to forward slashes
shoulderPath = shoulderPath.replace('\\', '/')

parts = input("Please input the direct path to the folder containing particle folders: ")
parts = parts.replace('\\', '/')

bone = input("Enter 1 for scapulae, 2 for humerus, or 3 for clavicle: ")
if bone == '1':
    parts=parts+"/all-scaps_particles/"
if bone == '2':
    parts=parts+"/all-hums_particles/"
if bone == '3':
    parts=parts+"/all-clav_particles/"
    
# Generate a list of paths to each group of .stl files
dirs = get_mvt_dirs(shoulderPath, study_list)
patientCount = len(dirs)

for count in range(0, patientCount-1):
    # Change the cwd to the path to each group of .stls
    os.chdir(dirs[count])
    #print(dirs[count])
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
                shutil.copy(file2,dirs[count])
                os.replace(dirs[count] +"/"+ file, "sca.kp")
    elif bone == '2':
        for file in glob.glob('*hum.stl'):
            file = os.path.splitext(file)[0]
            #print(file)
            file = file+"_groomed_local.particles"
            #print(file)
            # print(parts+file)
            for file2 in glob.glob(parts+file):
                # print(file2)
                shutil.copy(file2,dirs[count])
                os.replace(dirs[count] +"/"+ file, "hum.kp")
    elif bone == '3':
        for file in glob.glob('*cla.stl'):
            file = os.path.splitext(file)[0]
            #print(file)
            file = file+"_groomed_local.particles"
            #print(file)
            # print(parts+file)
            for file2 in glob.glob(parts+file):
                # print(file2)
                shutil.copy(file2,dirs[count])
                os.replace(dirs[count] +"/"+ file, "cla.kp")
    
print("Particles converted successfully.\n.kp and .jts files are located in each movement directory, for each side (L and R) that exists per patient.")
    
            