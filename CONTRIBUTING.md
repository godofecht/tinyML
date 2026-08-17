# Contributing to TinyML

Thanks for improving TinyML. Changes should be narrowly scoped, reproducible and accompanied by the smallest useful test coverage.

## Before opening a pull request

Build and run the registered test suite:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

If a change affects wall-clock performance, benchmark it on controlled hardware and include the hardware, compiler, build type and command used. Do not use shared GitHub-hosted runners as benchmark evidence.

Keep public API changes documented. New algorithms should include a focused example or test showing the intended behaviour. Avoid unrelated formatting or generated-file churn in functional pull requests.

## Pull requests

Describe the problem, the approach, how it was verified and any compatibility or licensing implications. CI must pass before merge.

## Licensing

Read `LICENSING.md` before adding new public headers or dependencies. Contributions must preserve the documented boundary between the MIT core and the separately licensed extended model library.
