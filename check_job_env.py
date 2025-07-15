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