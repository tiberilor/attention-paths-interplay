#!python3
# from https://www.osc.edu/book/export/html/4046
import argparse
import os
import csv, subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Automatically submit jobs using a csv file")

    parser.add_argument('--job_name', type=str, default="sncf")
    parser.add_argument('--jobscript', help="job script to use")
    parser.add_argument('--config', help="csv parameter file to use")
    parser.add_argument('-t','--test',action='store_true',help="test script without submitting jobs")

    args = parser.parse_args()

    with open(args.config, mode='r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip the first row
        for job in reader:
            submit_command = (
                f"sbatch --job-name={args.job_name} " +
                "--export=ALL,num_samples={0},model_width={1},temperature={2} ".format(*job) + args.jobscript)
            if args.test:
                print(submit_command)
            else:
                exit_status = subprocess.call(submit_command,shell=True)
                # Check to make sure the job submitted
                if exit_status == 1:
                    print("Job {0} failed to submit".format(submit_command))

    print("Done submitting jobs")

if __name__ == "__main__":
    main()
