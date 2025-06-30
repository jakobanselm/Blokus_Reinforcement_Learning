import importlib
import sys


def read_requirements(path="requirements.txt"):
    packages = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # remove version specifiers and extras
                pkg = line.split('==')[0].split('>=')[0].split('<=')[0]
                pkg = pkg.split('[')[0]
                if pkg:
                    packages.append(pkg)
    except FileNotFoundError:
        pass
    return packages


def main():
    pkgs = set(read_requirements())
    # Explicitly check important packages
    pkgs.update([
        "gymnasium",
        "ray.rllib",
        "env.blokus_env_multi_agent_ray_rllib",
    ])

    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
        except Exception as e:
            print(f"Fehler beim Import von {pkg}: {e}")
            sys.exit(1)
    print("\u2713 Dependencies OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
