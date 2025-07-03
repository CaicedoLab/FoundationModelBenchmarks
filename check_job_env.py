import subprocess

def main():
    result = subprocess.run(
                    f"condor_history jgpeters3 -limit 300",
                    shell=True,
                    capture_output=True,
                    check=True,
                    text=True
                )
    results = result.stdout.split('\n')
    for result in results[1:]:
        job_id = result.split()[0]
        result = subprocess.run(
                    f"condor_history -l {job_id} | grep MachineAttrMachine0",
                    shell=True,
                    capture_output=True,
                    check=True,
                    text=True
                )
        print(result.stdout.strip(), job_id)

if __name__ == "__main__":
    main()