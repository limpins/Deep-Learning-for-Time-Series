# How to build pytorch1.0 on Windows
## Step 1
[Install Microsoft Visual Studio 2017](https://visualstudio.microsoft.com/zh-hans/downloads/?rr=https%3A%2F%2Fwww.google.com.hk%2F)

## Step 2
**Switch to pytorch dirtory firstly.**

```bat
set "VS150COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build"
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64
set DISTUTILS_USE_SDK=1
call "%VS150COMNTOOLS%\vcvarsall.bat" x64
```

## Step 3
```bat
python setup.py install
```

### Note
1. Visual Studio2017 15.8.9
2. CUDA 9.0
3. Cudnn 7
4. Python 3.5
