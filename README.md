# The Memclave CI-switch

The CI-switch is responsible for instantiating the First-Stage-Loader on each DPU and
seed it with a random value. The FSL then generates keys, wraps the system subkernels with
the system key and instantiates the TL in turn. Then the MSG subkernel signals that it is
ready for communication, after the TL loaded and verified it.

During its lifetime, the CI-switch then facilitates all communication between the guest OS
and the DPU ranks. For that it creates a UNIX socket `/tmp/vci.sock`, which QEMU running
the Guest OS connects to. MRAM accesses are not performed via the UNIX socket, for these
QEMU directly maps the required regions into the Guest OS.

**Building the CI-switch**

The CI-switch depends on the FSL code, which depends on the TL code and all of the system
subkernel code. All of this is handled transparently via git submodules and cmake. Memclave
uses a customized version of UPMEM's LLVM port, which we ship using a docker container. The
Dockerfile for this container is in the `ime` repository and not here. We recommend building
the CI-switch within the container. The container build itself takes some time because it
has to recompile LLVM with our patches.

First, pull the docker container from the relevant registry. You may also build the container
present in the `ime` repository. So either run
```bash
docker pull ghcr.io/deinernstjetzt/upmem:latest
```
or alternatively
```bash
docker build -it memclave/memclave .
```
in the `ime` repository. This build will take some time, because it has to recompile LLVM using
our custom patches. The build now pulls dependencies from archive.org due to some issues with
availability of UPMEM's packages.

After building or pulling the container, you may launch it. If you need ssh for cloning git
repositories, we suggest forwarding the ssg-agent to it. This is optional though:
```bash
docker run -v ${SSH_AUTH_SOCK}:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent -it ghcr.io/deinernstjetzt/upmem:latest bash
```

In the container, you then pull the ci-switch repository using `git`. All of the `ci-switch` dependencies can
be pulled using git submodules.
```bash
git clone git@projects.cispa.saarland:fabian.van-rissenbeck/ci-switch.git
git submodule update --init --recursive
```

Then configure and build the CI-switch using `cmake`:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build --target ci-switch
```

**Using the CI-switch**

The CI-switch supports two command line options:
+ `--nr-ranks=` Specify how many ranks should be allocated. Any value between 1 and 40 should be valid. The upper
  limit depends on your UPMEM system.
+ `--exact-rank=` Specify, that you wish to allocate exactly 1 rank with a specific number. This cannot be used together
  with the `--nr-ranks` option.

Once started, the CI-switch will report faults on all DPUs. This is the intended behavior of the CI-switch, we use
UPMEM's fault mechanism for signaling purposes. These faults can savely be ignored. If there are no faults, something
went wrong.