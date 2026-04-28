import sys


def main() -> None:
    argv = sys.argv[1:]
    if "--mode" in argv:
        mode_idx = argv.index("--mode")
        mode = argv[mode_idx + 1] if mode_idx + 1 < len(argv) else None
        if mode == "compare":
            sys.argv = [sys.argv[0]] + argv[:mode_idx] + argv[mode_idx + 2 :]
            from evaluation.compare_suite import main as compare_main

            compare_main()
            return

    from evaluation.run_serialized_sin_evaluations import main as single_main

    single_main()


if __name__ == "__main__":
    main()
