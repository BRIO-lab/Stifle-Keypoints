import os
import csv
import glob

# Simple script to create csv files. Must be placed in root of Shoulder-Keypoint, with data in the "kp_data" folder.
# Outputs to a folder named "csv_data"

def writecsv(bone):
    # Check if the folder exists. If not, create it. 
    if os.path.isdir("csv_data") == False:
        os.mkdir("csv_data")
    with open("csv_data/"+bone+".csv", 'w', newline='') as c:
        csvFile = csv.writer(c)

        # Create header for the csv, used in FeaturePointDataset for data loading and processing
        header = ["grid", "keypoints"]
        os.chdir("kp_data")
        # Append the header to the top of the file
        csvFile.writerow(header)
        # Write to the csv
        for file in glob.glob("*"+bone+"*.txt"):
            f = open(file, 'r')
            # Save the file name to column 1
            d = [f.readline()]
            # Create an empty string to append the keypoints to, so they all end up in one column
            s = ""
            # Writes to the csv and creates a new row when the next image name is encountered
            for line in f:
                if "tif" in line:
                    s = s[:-1]
                    # Append the list of keypoints to the dictionary, and write to the csv
                    d.append(s)
                    csvFile.writerow(d)
                    # Clear the dict and string to prep for a new set of keypoints
                    s = ""
                    d = [line]
                else:
                    s += line + ","
                
        os.chdir(pwd)

# This is done because the createcsv script is in the "scripts" folder. 
# If you are running this in root, comment the next two lines out
os.chdir('../')       
pwd = os.getcwd()

writecsv("cla")
writecsv("hum")
writecsv("sca")