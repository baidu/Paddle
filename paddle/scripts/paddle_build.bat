@ECHO OFF
rem Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem =================================================
rem       Paddle CI Task On Windows Platform
rem =================================================

@ECHO ON
setlocal enabledelayedexpansion

rem -------clean up environment-----------
set work_dir=%cd%
if not defined cache_dir set cache_dir=%work_dir:Paddle=cache%
if not exist %cache_dir%\tools (
    cd /d %cache_dir%
    python -m pip install wget
    python -c "import wget;wget.download('https://paddle-ci.gz.bcebos.com/window_requirement/tools.zip')"
    tar xf tools.zip
    cd /d %work_dir%
)
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im ninja.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im python.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im eager_generator.exe /t 2>NUL
taskkill /f /im eager_legacy_op_function_generator.exe /t 2>NUL
wmic process where name="eager_generator.exe" call terminate 2>NUL
wmic process where name="eager_legacy_op_function_generator.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL
wmic process where name="python.exe" call terminate 2>NUL

rem variable to control building process
if not defined GENERATOR set GENERATOR="Visual Studio 15 2017 Win64"
if not defined WITH_TENSORRT set WITH_TENSORRT=ON
if not defined TENSORRT_ROOT set TENSORRT_ROOT=D:/TensorRT
if not defined WITH_GPU set WITH_GPU=ON
if not defined WITH_MKL set WITH_MKL=ON
if not defined WITH_AVX set WITH_AVX=ON
if not defined WITH_TESTING set WITH_TESTING=ON
if not defined MSVC_STATIC_CRT set MSVC_STATIC_CRT=ON
if not defined WITH_PYTHON set WITH_PYTHON=ON
if not defined ON_INFER set ON_INFER=ON
if not defined WITH_ONNXRUNTIME set WITH_ONNXRUNTIME=OFF
if not defined WITH_INFERENCE_API_TEST set WITH_INFERENCE_API_TEST=ON
if not defined WITH_STATIC_LIB set WITH_STATIC_LIB=ON
if not defined WITH_UNITY_BUILD set WITH_UNITY_BUILD=OFF
if not defined NEW_RELEASE_ALL set NEW_RELEASE_ALL=ON
if not defined NEW_RELEASE_PYPI set NEW_RELEASE_PYPI=OFF
if not defined NEW_RELEASE_JIT set NEW_RELEASE_JIT=OFF
if not defined WITH_CPP_TEST set WITH_CPP_TEST=ON
if not defined WITH_NIGHTLY_BUILD set WITH_NIGHTLY_BUILD=OFF

rem variable to control pipeline process
if not defined WITH_TPCACHE set WITH_TPCACHE=OFF
if not defined WITH_CACHE set WITH_CACHE=OFF
if not defined WITH_SCCACHE set WITH_SCCACHE=OFF
if not defined INFERENCE_DEMO_INSTALL_DIR set INFERENCE_DEMO_INSTALL_DIR=%cache_dir:\=/%/inference_demo
if not defined LOG_LEVEL set LOG_LEVEL=normal
if not defined PRECISION_TEST set PRECISION_TEST=OFF
if not defined WIN_UNITTEST_LEVEL set WIN_UNITTEST_LEVEL=2
rem LEVEL 0: For unittests unrelated to CUDA/TRT or unittests without GPU memory, only run on
rem          PR-CI-Windows-Infernece(CUDA 11.2), skip them on PR-CI-Windows(CUDA 12.0)
rem LEVEL 1: For unittests unrelated to CUDA/TRT, only run on PR-CI-Windows-Infernece(CUDA 11.2),
rem          skip them on PR-CI-Windows(CUDA 12.0)
rem LEVEL 2: run all test
if not defined NIGHTLY_MODE set NIGHTLY_MODE=OFF
if not defined retry_times set retry_times=1
if not defined PYTHON_ROOT set PYTHON_ROOT=C:\Python38
if not defined BUILD_DIR set BUILD_DIR=build
if not defined TEST_INFERENCE set TEST_INFERENCE=ON
if not defined WITH_PIP_CUDA_LIBRARIES set WITH_PIP_CUDA_LIBRARIES=OFF

set task_name=%1
set UPLOAD_TP_FILE=OFF
set UPLOAD_TP_CODE=OFF

set error_code=0
type %cache_dir%\error_code.txt

rem ------initialize set git config------
git config --global core.longpaths true

rem ------initialize the python environment------
set PYTHON_VENV_ROOT=%cache_dir%\python_venv
set PYTHON_EXECUTABLE=!PYTHON_VENV_ROOT!\Scripts\python.exe
%PYTHON_ROOT%\python.exe -m venv --clear !PYTHON_VENV_ROOT!
call !PYTHON_VENV_ROOT!\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo activate python virtual environment failed!
    exit /b 5
)

if "%WITH_PYTHON%" == "ON" (
    where python
    where pip
    python -m pip install --upgrade pip
    python -m pip install -r %work_dir%\paddle\scripts\compile_requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo pip install compile_requirements.txt failed!
        exit /b 5
    )
    python -m pip install -r %work_dir%\python\requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo pip install requirements.txt failed!
        exit /b 5
    )
)

rem -------Caching strategy 1: keep build directory for incremental compilation-----------
if "%WITH_CACHE%"=="OFF" (
    rmdir %BUILD_DIR% /s/q
    goto :mkbuild
)

rmdir %BUILD_DIR%\python /s/q
rmdir %BUILD_DIR%\paddle_install_dir /s/q
rmdir %BUILD_DIR%\paddle_inference_install_dir /s/q
rmdir %BUILD_DIR%\paddle_inference_c_install_dir /s/q
del %BUILD_DIR%\CMakeCache.txt

: set /p error_code=< %cache_dir%\error_code.txt
if %error_code% NEQ 0 (
    rmdir %BUILD_DIR% /s/q
    goto :mkbuild
)

setlocal enabledelayedexpansion
git show-ref --verify --quiet refs/heads/last_pr
if %ERRORLEVEL% EQU 0 (
    git diff HEAD last_pr --stat --name-only
    git diff HEAD last_pr --stat --name-only | findstr "setup.py.in"
    if !ERRORLEVEL! EQU 0 (
        rmdir %BUILD_DIR% /s/q
    )
    git branch -D last_pr
    git branch last_pr
) else (
    rmdir %BUILD_DIR% /s/q
    git branch last_pr
)

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set datetime=%%#
set day_now=%datetime:~6,2%
set day_before=-1
set /p day_before=< %cache_dir%\day.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %cache_dir%\day.txt
    type %cache_dir%\day.txt
    rmdir %BUILD_DIR% /s/q
    goto :mkbuild
)

:mkbuild
if not exist %BUILD_DIR% (
    echo Windows build cache FALSE
    set Windows_Build_Cache=FALSE
    mkdir %BUILD_DIR%
) else (
    echo Windows build cache TRUE
    set Windows_Build_Cache=TRUE
)
echo ipipe_log_param_Windows_Build_Cache: %Windows_Build_Cache%
cd /d %BUILD_DIR%
dir .
dir %cache_dir%
rem -------Caching strategy 1: End --------------------------------


rem -------Caching strategy 2: sccache decorate compiler-----------
if not defined SCCACHE_ROOT set SCCACHE_ROOT=D:\sccache
set PATH=%SCCACHE_ROOT%;%PATH%
if "%WITH_SCCACHE%"=="ON" (
    cmd /C sccache -V || call :install_sccache

    sccache --stop-server 2> NUL
    del %SCCACHE_ROOT%\sccache_log.txt

    :: Locally storage on windows
    if not exist %SCCACHE_ROOT% mkdir %SCCACHE_ROOT%
    set SCCACHE_DIR=%SCCACHE_ROOT%\.cache

    :: Sccache will shut down if a source file takes more than 10 mins to compile
    set SCCACHE_IDLE_TIMEOUT=0
    set SCCACHE_CACHE_SIZE=100G
    set SCCACHE_ERROR_LOG=%SCCACHE_ROOT%\sccache_log.txt
    set SCCACHE_LOG=quiet

    :: Distributed storage on windows
    set SCCACHE_ENDPOINT=s3.bj.bcebos.com
    set SCCACHE_BUCKET=paddle-windows
    set SCCACHE_S3_KEY_PREFIX=sccache/
    set SCCACHE_S3_USE_SSL=true

    sccache --start-server
    sccache -z
    goto :CASE_%1
) else (
    del %SCCACHE_ROOT%\sccache.exe 2> NUL
    goto :CASE_%1
)

:install_sccache
echo There is not sccache in this PC, will install sccache.
echo Download package from https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe
python -c "import wget;wget.download('https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe')"
xcopy sccache.exe %SCCACHE_ROOT%\ /Y
del sccache.exe
goto:eof
rem -------Caching strategy 2: End --------------------------------

echo "Usage: paddle_build.bat [OPTION]"
echo "OPTION:"
echo "wincheck_mkl: run Windows MKL/GPU PR CI tasks on Windows"
echo "wincheck_openbals: run Windows OPENBLAS/CPU PR CI tasks on Windows"
echo "build_avx_whl: build Windows avx whl package on Windows"
echo "build_no_avx_whl: build Windows no avx whl package on Windows"
echo "build_inference_lib: build Windows inference library on Windows"
exit /b 1

rem ------PR CI windows check for MKL/GPU----------
:CASE_wincheck_mkl
set WITH_MKL=ON
set WITH_GPU=ON
set WITH_AVX=ON
set MSVC_STATIC_CRT=OFF
set ON_INFER=ON
set WITH_TENSORRT=ON
set WITH_INFERENCE_API_TEST=OFF
set WIN_UNITTEST_LEVEL=0
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_package || goto test_whl_package_error
call :test_unit || goto test_unit_error
call :test_inference || goto test_inference_error
goto:success

rem ------PR CI windows check for OPENBLAS/CPU------
:CASE_wincheck_openblas
set WITH_MKL=OFF
set WITH_GPU=OFF
set WITH_AVX=OFF
set MSVC_STATIC_CRT=ON
set ON_INFER=ON
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_package || goto test_whl_package_error
call :test_unit || goto test_unit_error
goto:success

rem ------PR CI windows check for unittests and inference in CUDA11-MKL-AVX----------
:CASE_wincheck_inference
set WITH_MKL=ON
set WITH_GPU=ON
set WITH_AVX=ON
set MSVC_STATIC_CRT=ON
set ON_INFER=ON
set WITH_TENSORRT=ON
set WITH_INFERENCE_API_TEST=ON
set WITH_ONNXRUNTIME=ON
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=Auto

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_package || goto test_whl_package_error
call :test_unit || goto test_unit_error
::call :test_inference || goto test_inference_error
::call :test_inference_ut || goto test_inference_ut_error
::call :check_change_of_unittest || goto check_change_of_unittest_error
goto:success

rem ------Build windows avx whl package------
:CASE_build_avx_whl
set WITH_AVX=ON
set ON_INFER=ON
set WITH_PIP_CUDA_LIBRARIES=ON
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=All

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_package || goto test_whl_package_error
goto:success

rem ------Build windows no-avx whl package------
:CASE_build_no_avx_whl
set WITH_AVX=OFF
set ON_INFER=ON
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=All

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_package || goto test_whl_package_error
goto:success

rem ------Build windows inference library------
:CASE_build_inference_lib
set ON_INFER=ON
set WITH_PYTHON=OFF
if not defined CUDA_ARCH_NAME set CUDA_ARCH_NAME=All

python %work_dir%\tools\remove_grad_op_and_kernel.py
if %errorlevel% NEQ 0 exit /b 1

call :cmake || goto cmake_error
call :build || goto build_error
if "%TEST_INFERENCE%"=="ON" call :test_inference
if %errorlevel% NEQ 0 set error_code=%errorlevel%
if "%TEST_INFERENCE%"=="ON" call :test_inference_ut
if %errorlevel% NEQ 0 set error_code=%errorlevel%

call :zip_cc_file || goto zip_cc_file_error
call :zip_c_file || goto zip_c_file_error
if %error_code% NEQ 0 goto test_inference_error
goto:success

rem "Other configurations are added here"
rem :CASE_wincheck_others
rem call ...


rem ---------------------------------------------------------------------------------------------
:cmake
@ECHO OFF
echo    ========================================
echo    Step 1. Cmake ...
echo    ========================================

rem set vs language to english to block showIncludes, this need vs has installed English language package.
set VSLANG=1033
rem Configure the environment for 64-bit builds. 'DISTUTILS_USE_SDK' indicates that the user has selected the compiler.
if not defined vcvars64_dir set vcvars64_dir="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
call %vcvars64_dir%

set DISTUTILS_USE_SDK=1
rem Windows 10 Kit bin dir
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.17763.0\x64;%PATH%
rem Use 64-bit ToolSet to compile
set PreferredToolArchitecture=x64

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%

if not defined CUDA_TOOLKIT_ROOT_DIR set CUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2
set PATH=%TENSORRT_ROOT:/=\%\lib;%CUDA_TOOLKIT_ROOT_DIR:/=\%\bin;%CUDA_TOOLKIT_ROOT_DIR:/=\%\libnvvp;%PATH%

@ECHO ON
if "%WITH_GPU%"=="ON" (
    set cuda_version=%CUDA_TOOLKIT_ROOT_DIR:~-4%
    if "!cuda_version!"=="12.0" (
        set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64;%PATH%"
    )
)
echo %PATH%
rem CUDA_TOOLKIT_ROOT_DIR in cmake must use / rather than \
set TENSORRT_ROOT=%TENSORRT_ROOT:\=/%
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR:\=/%

rem install ninja if GENERATOR is Ninja
if %GENERATOR% == "Ninja" (
    rem Set the default generator for cmake to Ninja
    setx CMAKE_GENERATOR Ninja
    pip install ninja
    if %errorlevel% NEQ 0 (
        echo pip install ninja failed!
        exit /b 5
    )
)

rem ------show summary of current GPU environment----------
cmake --version
if "%WITH_GPU%"=="ON" (
    nvcc --version
    nvidia-smi 2>NUL
)

rem ------set third_party cache dir------

if "%WITH_TPCACHE%"=="OFF" (
    set THIRD_PARTY_PATH=%work_dir:\=/%/%BUILD_DIR%/third_party
    goto :cmake_impl
)

rem clear third party cache every ten days
for /F %%# in ('wmic os get localdatetime^|findstr 20') do set datetime=%%#
set day_now=%datetime:~6,2%
set day_before=-1
set /p day_before=< %cache_dir%\day_third_party.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %cache_dir%\day_third_party.txt
    type %cache_dir%\day_third_party.txt
    if %day_now% EQU 21 (
        rmdir %cache_dir%\third_party /s/q
    )
    if %day_now% EQU 11 (
        rmdir %cache_dir%\third_party /s/q
    )
    if %day_now% EQU 01 (
        rmdir %cache_dir%\third_party /s/q
    )
)

echo set -ex > cache.sh
echo md5_content=$(cat %work_dir:\=/%/cmake/external/*.cmake^|md5sum^|awk '{print $1}')$(git submodule status^|md5sum^|awk '{print $1}')>>cache.sh
echo echo ${md5_content}^>md5.txt>>cache.sh

%cache_dir%\tools\busybox64.exe cat cache.sh
%cache_dir%\tools\busybox64.exe bash cache.sh

set /p md5=< md5.txt
if "%WITH_GPU%"=="ON" (
    set cuda_version=%CUDA_TOOLKIT_ROOT_DIR:~-4%
    set sub_dir=cuda!cuda_version:.=!
) else (
    set sub_dir=cpu
)

@ECHO ON
cd /d %work_dir%
python -c "import wget;wget.download('https://paddle-windows.bj.bcebos.com/third_party_code/%sub_dir%/%md5%.tar.gz')" 2>nul
if !ERRORLEVEL! EQU 0 (
    echo Getting source code of third party : extracting ...
    tar -xf %md5%.tar.gz
    del %md5%.tar.gz
    if !errorlevel! EQU 0 (
        echo Getting source code of third party : successful
    )
) else (
    git submodule update --init --recursive
    if !errorlevel! EQU 0 (
        set UPLOAD_TP_CODE=ON
    )
)
if "%UPLOAD_TP_CODE%"=="ON" (
    set BCE_FILE=%cache_dir%\bce-python-sdk-new\BosClient.py
    echo Uploading source code of third_party: checking bce ...
    if not exist %cache_dir%\bce-python-sdk-new (
        echo There is no bce in this PC, will install bce.
        cd /d %cache_dir%
        echo Download package from https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz
        python -c "import wget;wget.download('https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz')"
        python -c "import shutil;shutil.unpack_archive('bos_new.tar.gz', extract_dir='./bce-python-sdk-new',format='gztar')"
    )
    python -m pip install pycryptodome
    python -m pip install bce-python-sdk==0.8.74
    if !errorlevel! EQU 0 (
        cd /d %work_dir%
        echo Uploading source code of third party: compressing ...
        tar -zcf %md5%.tar.gz ./third_party ./.git/modules
        if !errorlevel! EQU 0 (
            echo Uploading source code of third party: uploading ...
            python !BCE_FILE! %md5%.tar.gz paddle-windows/third_party_code/%sub_dir% 1>nul
            if !errorlevel! EQU 0 (
                echo Upload source code of third party %md5% to bos paddle-windows/third_party_code/%sub_dir% successfully.
            ) else (
                echo Failed upload source code of third party to bos, reason: upload failed.
            )
        ) else (
            echo Failed upload source code of third party to bos, reason: compress failed.
        )
        del %md5%.tar.gz
    ) else (
        echo Failed upload source code of third party to bos, reason: install bce failed.
    )
)

set THIRD_PARTY_HOME=%cache_dir:\=/%/third_party/%sub_dir%
set THIRD_PARTY_PATH=%THIRD_PARTY_HOME%/%md5%

echo %task_name%|findstr build >nul && (
    echo %task_name% is a whl-build task, will only reuse local third_party cache.
    goto :cmake_impl
) || (
    echo %task_name% is a PR-CI-Windows task, will try to reuse bos and local third_party cache both.
)

:cmake_impl
if "%WITH_TESTING%"=="ON" (
    cd /d %work_dir%\%BUILD_DIR%
    rem whether to run cpp test
    python -m pip install PyGithub
    python %work_dir%\tools\check_only_change_python_files.py
    if exist %work_dir%\%BUILD_DIR%\only_change_python_file.txt set WITH_CPP_TEST=OFF
    echo WITH_CPP_TEST: %WITH_CPP_TEST%
)
cd /d %work_dir%\%BUILD_DIR%
echo cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUDA_ARCH_BIN=%CUDA_ARCH_BIN% -DCUB_PATH=%THIRD_PARTY_HOME%/cub ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DNEW_RELEASE_ALL=%NEW_RELEASE_ALL% -DNEW_RELEASE_PYPI=%NEW_RELEASE_PYPI% ^
-DNEW_RELEASE_JIT=%NEW_RELEASE_JIT% -DWITH_ONNXRUNTIME=%WITH_ONNXRUNTIME% -DWITH_CPP_TEST=%WITH_CPP_TEST% ^
-DWIN_UNITTEST_LEVEL=%WIN_UNITTEST_LEVEL% -DWITH_NIGHTLY_BUILD=%WITH_NIGHTLY_BUILD% -DWITH_PIP_CUDA_LIBRARIES=%WITH_PIP_CUDA_LIBRARIES% ^
-DWITH_SCCACHE=%WITH_SCCACHE% >> %work_dir%\win_cmake.sh

echo cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUDA_ARCH_BIN=%CUDA_ARCH_BIN% -DCUB_PATH=%THIRD_PARTY_HOME%/cub ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DNEW_RELEASE_ALL=%NEW_RELEASE_ALL% -DNEW_RELEASE_PYPI=%NEW_RELEASE_PYPI% ^
-DNEW_RELEASE_JIT=%NEW_RELEASE_JIT% -DWITH_ONNXRUNTIME=%WITH_ONNXRUNTIME% -DWITH_CPP_TEST=%WITH_CPP_TEST% ^
-DWIN_UNITTEST_LEVEL=%WIN_UNITTEST_LEVEL% -DWITH_NIGHTLY_BUILD=%WITH_NIGHTLY_BUILD% -DWITH_PIP_CUDA_LIBRARIES=%WITH_PIP_CUDA_LIBRARIES% ^
-DWITH_SCCACHE=%WITH_SCCACHE%>> %work_dir%\win_cmake.sh

cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUDA_ARCH_BIN=%CUDA_ARCH_BIN% -DCUB_PATH=%THIRD_PARTY_HOME%/cub ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DNEW_RELEASE_ALL=%NEW_RELEASE_ALL% -DNEW_RELEASE_PYPI=%NEW_RELEASE_PYPI% ^
-DNEW_RELEASE_JIT=%NEW_RELEASE_JIT% -DWITH_ONNXRUNTIME=%WITH_ONNXRUNTIME% -DWITH_CPP_TEST=%WITH_CPP_TEST% ^
-DWIN_UNITTEST_LEVEL=%WIN_UNITTEST_LEVEL% -DWITH_NIGHTLY_BUILD=%WITH_NIGHTLY_BUILD% -DWITH_PIP_CUDA_LIBRARIES=%WITH_PIP_CUDA_LIBRARIES% ^
-DWITH_SCCACHE=%WITH_SCCACHE%
goto:eof

:cmake_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Cmake failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:build
@ECHO OFF
echo    ========================================
echo    Step 2. Build Paddle ...
echo    ========================================

for /F %%# in ('wmic cpu get NumberOfLogicalProcessors^|findstr [0-9]') do set /a PARALLEL_PROJECT_COUNT=%%#*4/5
echo "PARALLEL PROJECT COUNT is %PARALLEL_PROJECT_COUNT%"

set build_times=1
rem MSbuild will build third_party first to improve compiler stability.
if NOT %GENERATOR% == "Ninja" (
    goto :build_tp
) else (
    goto :build_paddle
)

:build_tp
echo Build third_party the %build_times% time:
if %GENERATOR% == "Ninja" (
    ninja third_party
) else (
    MSBuild /m /p:PreferredToolArchitecture=x64 /p:Configuration=Release /verbosity:%LOG_LEVEL% third_party.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build third_party failed, will retry!
        goto :build_tp
    )
)
echo Build third_party successfully!

set build_times=1

:build_paddle
:: reset clcache zero stats for collect PR's actual hit rate
rem clcache.exe -z

rem -------clean up environment again-----------
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im eager_generator.exe /t 2>NUL
taskkill /f /im eager_legacy_op_function_generator.exe /t 2>NUL
wmic process where name="eager_generator.exe" call terminate 2>NUL
wmic process where name="eager_legacy_op_function_generator.exe" call terminate 2>NUL
wmic process where name="cmake.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL

if "%WITH_TESTING%"=="ON" (
    for /F "tokens=1 delims= " %%# in ('tasklist ^| findstr /i test') do taskkill /f /im %%# /t
)

echo Build Paddle the %build_times% time:
if %GENERATOR% == "Ninja" (
    ninja all
) else (
    MSBuild /m:%PARALLEL_PROJECT_COUNT% /p:PreferredToolArchitecture=x64 /p:TrackFileAccess=false /p:Configuration=Release /verbosity:%LOG_LEVEL% ALL_BUILD.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build Paddle failed, will retry!
        goto :build_paddle
    )
)

echo Build Paddle successfully!
echo 0 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt

:: ci will collect sccache hit rate
if "%WITH_SCCACHE%"=="ON" (
    call :collect_sccache_hits
)

goto:eof

:build_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Build Paddle failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:test_whl_package
@ECHO OFF
echo    ========================================
echo    Step 3. Test pip install whl package ...
echo    ========================================

setlocal enabledelayedexpansion

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "Build"

rem Record the exact size of dll and whl files and save them to disk D
set dll_file=%cd%\paddle\fluid\pybind\libpaddle.dll
for /F "tokens=1-5" %%a in ('dir "%dll_file%"') do (
    echo "%%e" | findstr  "libpaddle.dll" >nul
    if !errorlevel! equ 0 (
        set dllsize=%%d
        goto dll_break
    )
    echo "%%d" | findstr  "libpaddle.dll" >nul
    if !errorlevel! equ 0 (
        set dllsize=%%c
        goto dll_break
    )
)
:dll_break
echo Windows libpaddle.dll Size: %dllsize% bytes
set dllsize_folder=D:\record\dll_size
if not exist "%dllsize_folder%" (
    mkdir %dllsize_folder%
)
if exist "%dllsize_folder%\%AGILE_PULL_ID%.txt" (
    del "%dllsize_folder%\%AGILE_PULL_ID%.txt"
)
echo %dllsize% > %dllsize_folder%\%AGILE_PULL_ID%.txt

set whl_folder=%cd%\python\dist
for /F "tokens=1-5" %%a in ('dir "%whl_folder%"') do (
    echo "%%e" | findstr  ".whl" >nul
    if !errorlevel! equ 0 (
        set whlsize=%%d
        goto whl_break
    )
    echo "%%d" | findstr  ".whl" >nul
    if !errorlevel! equ 0 (
        set whlsize=%%c
        goto whl_break
    )
)
:whl_break
echo Windows PR whl Size: %whlsize% bytes
echo ipipe_log_param_Windows_PR_whl_Size: %whlsize% bytes
set whlsize_folder=D:\record\whl_size
if not exist "%whlsize_folder%" (
    mkdir %whlsize_folder%
)
if exist "%whlsize_folder%\%AGILE_PULL_ID%.txt" (
    del "%whlsize_folder%\%AGILE_PULL_ID%.txt"
)
echo %whlsize% > %whlsize_folder%\%AGILE_PULL_ID%.txt

dir /s /b python\dist\*.whl > whl_file.txt
set /p PADDLE_WHL_FILE_WIN=< whl_file.txt

@ECHO ON
pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu
pip install %PADDLE_WHL_FILE_WIN%
%PYTHON_ROOT%\python.exe -m pip uninstall -y paddlepaddle
%PYTHON_ROOT%\python.exe -m pip uninstall -y paddlepaddle-gpu
%PYTHON_ROOT%\python.exe -m pip install %PADDLE_WHL_FILE_WIN%

if %ERRORLEVEL% NEQ 0 (
    echo pip install whl package failed!
    exit /b 1
)

set CUDA_VISIBLE_DEVICES=0
python %work_dir%\paddle\scripts\installation_validate.py
goto:eof

:test_whl_package_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Test import paddle failed, will exit!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:test_unit
@ECHO ON
echo    ========================================
echo    Step 4. Running unit tests ...
echo    ========================================

pip install -r %work_dir%\python\unittest_py\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo pip install unittest requirements.txt failed!
    exit /b 5
)

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%

set FLAGS_call_stack_level=2
dir %THIRD_PARTY_PATH:/=\%\install\openblas\lib
dir %THIRD_PARTY_PATH:/=\%\install\openblas\bin
dir %THIRD_PARTY_PATH:/=\%\install\zlib\bin
dir %THIRD_PARTY_PATH:/=\%\install\mklml\lib
dir %THIRD_PARTY_PATH:/=\%\install\onednn\lib
dir %THIRD_PARTY_PATH:/=\%\install\warpctc\bin
dir %THIRD_PARTY_PATH:/=\%\install\onnxruntime\lib

set PATH=%THIRD_PARTY_PATH:/=\%\install\openblas\lib;%THIRD_PARTY_PATH:/=\%\install\openblas\bin;^
%THIRD_PARTY_PATH:/=\%\install\zlib\bin;%THIRD_PARTY_PATH:/=\%\install\mklml\lib;^
%THIRD_PARTY_PATH:/=\%\install\onednn\lib;%THIRD_PARTY_PATH:/=\%\install\warpctc\bin;^
%THIRD_PARTY_PATH:/=\%\install\onnxruntime\lib;%THIRD_PARTY_PATH:/=\%\install\paddle2onnx\lib;^
%work_dir%\%BUILD_DIR%\paddle\fluid\inference;%work_dir%\%BUILD_DIR%\paddle\fluid\pybind;%work_dir%\%BUILD_DIR%\paddle\fluid\inference\capi_exp;%work_dir%\%BUILD_DIR%\paddle\ir;^
%PATH%

REM TODO: make ut find .dll in install\onnxruntime\lib
if "%WITH_ONNXRUNTIME%"=="ON" (
    xcopy %THIRD_PARTY_PATH:/=\%\install\onnxruntime\lib\onnxruntime.dll %work_dir%\%BUILD_DIR%\paddle\fluid\inference\tests\api\ /Y
)

if "%WITH_GPU%"=="ON" (
    call:parallel_test_base_gpu
) else (
    call:parallel_test_base_cpu
)

set error_code=%ERRORLEVEL%

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"

if %error_code% NEQ 0 (
    exit /b 8
) else (
    goto:eof
)

:parallel_test_base_gpu
echo    ========================================
echo    Running GPU unit tests in parallel way ...
echo    ========================================

setlocal enabledelayedexpansion

:: set PATH=C:\Windows\System32;C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%
:: cmd /C nvidia-smi -L
:: if %errorlevel% NEQ 0 exit /b 8
:: for /F %%# in ('cmd /C nvidia-smi -L ^|find "GPU" /C') do set CUDA_DEVICE_COUNT=%%#
set CUDA_DEVICE_COUNT=1

:: For hypothesis tests(onednn op and inference pass), we set use 'ci' profile
set HYPOTHESIS_TEST_PROFILE=ci

%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE% %PRECISION_TEST% %WITH_GPU%

goto:eof

:parallel_test_base_cpu
echo    ========================================
echo    Running CPU unit tests in parallel way ...
echo    ========================================

:: For hypothesis tests(onednn op and inference pass), we set use 'ci' profile
set HYPOTHESIS_TEST_PROFILE=ci
%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE% %PRECISION_TEST% %WITH_GPU%

goto:eof

:test_unit_error
:: echo 8 > %cache_dir%\error_code.txt
:: type %cache_dir%\error_code.txt
echo Running unit tests failed, will exit!
exit /b 8

rem ---------------------------------------------------------------------------------------------
:test_inference
@ECHO OFF
echo    ========================================
echo    Step 5. Testing fluid library for inference ...
echo    ========================================

tree /F %cd%\paddle_inference_install_dir\paddle
%cache_dir%\tools\busybox64.exe du -h -d 0 %cd%\paddle_inference_install_dir > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do echo "Windows Paddle_Inference Size: !libsize_m!M"
for /F %%i in ("%libsize%") do echo ipipe_log_param_Windows_Paddle_Inference_Size: !libsize_m!M

cd /d %work_dir%\paddle\fluid\inference\api\demo_ci
%cache_dir%\tools\busybox64.exe bash run.sh %work_dir:\=/% %WITH_MKL% %WITH_GPU% %cache_dir:\=/%/inference_demo %WITH_TENSORRT% %TENSORRT_ROOT% %WITH_ONNXRUNTIME% %MSVC_STATIC_CRT% "%CUDA_TOOLKIT_ROOT_DIR%"

goto:eof

:test_inference_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo    ==========================================
echo    Testing inference library failed!
echo    ==========================================
exit /b 1

rem ---------------------------------------------------------------------------------------------
:check_change_of_unittest
@ECHO OFF
echo    ========================================
echo    Step 6. Check whether deleting a unit test ...
echo    ========================================

%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\check_change_of_unittest.sh

goto:eof

:check_change_of_unittest_error
exit /b 1

rem ---------------------------------------------------------------------------------------------
:test_inference_ut
@ECHO OFF
echo    ========================================
echo    Step 7. Testing fluid library with infer_ut for inference ...
echo    ========================================

cd /d %work_dir%\test\cpp\inference\infer_ut
%cache_dir%\tools\busybox64.exe bash run.sh %work_dir:\=/% %WITH_MKL% %WITH_GPU% %cache_dir:\=/%/inference_demo %TENSORRT_ROOT% %WITH_ONNXRUNTIME% %MSVC_STATIC_CRT% "%CUDA_TOOLKIT_ROOT_DIR%"
goto:eof

:test_inference_ut_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Testing fluid library with infer_ut for inference failed!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:zip_cc_file
cd /d %work_dir%\%BUILD_DIR%
tree /F %cd%\paddle_inference_install_dir\paddle
if exist paddle_inference.zip del paddle_inference.zip
python -c "import shutil;shutil.make_archive('paddle_inference', 'zip', root_dir='paddle_inference_install_dir')"
%cache_dir%\tools\busybox64.exe du -h -k paddle_inference.zip > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do (
    set /a libsize_m=%%i/1024
    echo "Windows Paddle_Inference ZIP Size: !libsize_m!M"
)
goto:eof

:zip_cc_file_error
echo Tar inference library failed!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:zip_c_file
cd /d %work_dir%\%BUILD_DIR%
tree /F %cd%\paddle_inference_c_install_dir\paddle
if exist paddle_inference_c.zip del paddle_inference_c.zip
python -c "import shutil;shutil.make_archive('paddle_inference_c', 'zip', root_dir='paddle_inference_c_install_dir')"
%cache_dir%\tools\busybox64.exe du -h -k paddle_inference_c.zip > lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do (
    set /a libsize_m=%%i/1024
    echo "Windows Paddle_Inference CAPI ZIP Size: !libsize_m!M"
)
goto:eof

:zip_c_file_error
echo Tar inference capi library failed!
exit /b 1

:timestamp
setlocal enabledelayedexpansion
@ECHO OFF
set start=%~1
set dd=%start:~2,2%
set /a dd=100%dd%%%100
set hh=%start:~4,2%
set /a hh=100%hh%%%100
set nn=%start:~6,2%
set /a nn=100%nn%%%100
set ss=%start:~8,2%
set /a ss=100%ss%%%100
set /a start_sec=dd*86400+hh*3600+nn*60+ss
echo %start_sec%

set end=%~2
set dd=%end:~2,2%
set /a dd=100%dd%%%100
if %start:~0,2% NEQ %end:~0,2% (
    set month_day=0
    for %%i in (01 03 05 07 08 10 12) DO if %%i EQU %start:~0,2% set month_day=31
    for %%i in (04 06 09 11) DO if %%i EQU %start:~0,2% set month_day=30
    for %%i in (02) DO if %%i EQU %start:~0,2% set month_day=28
    set /a dd=%dd%+!month_day!
)
set hh=%end:~4,2%
set /a hh=100%hh%%%100
set nn=%end:~6,2%
set /a nn=100%nn%%%100
set ss=%end:~8,2%
set /a ss=100%ss%%%100
set /a end_secs=dd*86400+hh*3600+nn*60+ss
set /a cost_secs=end_secs-start_sec
echo "Windows %~3 Time: %cost_secs%s"
set tempTaskName=%~3
echo ipipe_log_param_Windows_%tempTaskName: =_%_Time: %cost_secs%s
goto:eof


:collect_sccache_hits
sccache -s > sccache_summary.txt
echo    ========================================
echo    sccache statistical summary ...
echo    ========================================
type sccache_summary.txt
for /f "tokens=2,3" %%i in ('type sccache_summary.txt ^| findstr "requests hits" ^| findstr /V "executed C/C++ CUDA"') do set %%i=%%j
if %requests% EQU 0 (
    echo "sccache hit rate: 0%"
    echo ipipe_log_param_sccache_Hit_Hate: 0%
) else (
    set /a rate=!hits!*10000/!requests!
    echo "sccache hit rate: !rate:~0,-2!.!rate:~-2!%%"
    echo ipipe_log_param_sccache_Hit_Hate: !rate:~0,-2!.!rate:~-2!%%
)

goto:eof


rem ---------------------------------------------------------------------------------------------
:success
echo    ========================================
echo    Clean up environment  at the end ...
echo    ========================================
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im ninja.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im git.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im git-remote-https.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im python.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im eager_generator.exe /t 2>NUL
taskkill /f /im eager_legacy_op_function_generator.exe /t 2>NUL
wmic process where name="eager_generator.exe" call terminate 2>NUL
wmic process where name="eager_legacy_op_function_generator.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL
wmic process where name="python.exe" call terminate 2>NUL
if "%WITH_TESTING%"=="ON" (
    for /F "tokens=1 delims= " %%# in ('tasklist ^| findstr /i test') do taskkill /f /im %%# /t
)
echo Windows CI run successfully!
exit /b 0

ENDLOCAL
