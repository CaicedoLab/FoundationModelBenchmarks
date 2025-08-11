import subprocess

def main():
    result = subprocess.run(
                    f"condor_history jgpeters3 -limit 100",
                    shell=True,
                    capture_output=True,
                    check=True,
                    text=True
                )
    results = result.stdout.split('\n')[1:-1]
    for result in results[1:]:
        time_values = map(int, result.split()[4][2:].split(":"))
        total_time = sum(time_values)
        if total_time > 0:
            job_id = result.split()[0]
            try:
                result = subprocess.run(
                            f"condor_history -l {job_id}",
                            shell=True,
                            capture_output=True,
                            check=True,
                            text=True
                        )
                if "RUN_NAME=" in result.stdout.strip():
                    for line in  result.stdout.strip().splitlines():
                       if "MachineAttrMachine0" in line:
                           machine = line
                       elif "RUN_NAME=" in line:
                           run_name = line
                    
                    print(run_name, machine)
            except:
                continue

if __name__ == "__main__":
    main()