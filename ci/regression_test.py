import multiprocessing
import subprocess
import os
import time

from pathlib import Path


def run_cmd(cmd: Path, use_verify):
    dirpath = cmd.parent
    outpath = dirpath.joinpath("run.log")
    errpath = dirpath.joinpath("err.log")
    if use_verify:
        args = os.path.join("./", "verify")
    else:
        args = os.path.join("./", cmd.name)
    stdout = open(outpath, "w")
    stderr = open(errpath, "w")
    subprocess.check_call(
        args=args, cwd=dirpath, shell=True, stdout=stdout, stderr=stderr
    )


class TaskApp:
    def __init__(self, cmd, use_verify) -> None:
        self.cmd = cmd
        self.use_verify = use_verify
        self.time = 0
        self.state = "wait"


class TestManager:
    def __init__(self) -> None:
        self.tasks = {}
        self.complete_cnt = 0
        self.start_time = time.perf_counter()
        self.apps = {}
        self.pool = self.create_pool()

    def create_pool(self):
        cpu_num = multiprocessing.cpu_count()
        process_num = int(min(cpu_num / 2, max(cpu_num, 1)))
        pool = multiprocessing.Pool(processes=process_num)
        return pool

    def run(self):
        self.start_task()
        while True:
            running, successful, error = [], [], []
            for app_name, result in self.tasks.items():
                state = "error"
                try:
                    if result.successful():
                        state = "success"
                        successful.append(app_name)
                    else:
                        state = "error"
                        error.append(app_name)
                except:
                    state = "running"
                    self.apps[app_name].time = time.perf_counter() - self.start_time
                    running.append(app_name)
                self.apps[app_name].state = state

            self.print_table()
            if len(successful) + len(error) == self.complete_cnt:
                break
            time.sleep(1)
        self.pool.close()
        self.pool.join()

    def start_task(self):
        for name, app in self.apps.items():
            self.tasks[name] = self.pool.apply_async(
                func=run_cmd, args=(app.cmd, app.use_verify)
            )

    def add_test_app(self, cmd: Path, use_verify=False):
        self.complete_cnt += 1
        app_name = cmd.parent.name
        self.apps[app_name] = TaskApp(cmd, use_verify)

    def print_table(self):
        os.system("clear")
        print("{:^40}".format("Test Applications"))
        print("| {:<30} | {:<10} | {:<5} |".format("Applications", "State", "Time"))
        for app_name, app in self.apps.items():
            print(
                "| {:<30} | {:<10} | {:<5.2f} |".format(app_name, app.state, app.time)
            )


def polybench_app():
    polybench_root = Path("../benchmarks/polybench/CUDA")
    for dir in polybench_root.iterdir():
        app_path = dir.joinpath(dir.name)
        if app_path.exists():
            manager.add_test_app(app_path)
            print(app_path)


def basic_app():
    basic_root = Path("../benchmarks/basic")
    for dir in basic_root.iterdir():
        verify_path = dir.joinpath("verify")
        if verify_path.exists():
            manager.add_test_app(verify_path, True)
            print(verify_path)
            continue
        app_path = dir.joinpath(dir.name)
        if app_path.exists():
            manager.add_test_app(app_path)
            print(app_path)


if __name__ == "__main__":
    manager = TestManager()
    polybench_app()
    basic_app()
    manager.run()
