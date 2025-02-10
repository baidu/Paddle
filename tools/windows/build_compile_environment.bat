:: Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::
:: ===============================
:: Build Paddle compile environment
:: ===============================
:: Description:
::
::   Install compile environment for xly CI.
::
::   Include:
::     1. CMake 3.18.0
::     2. Git 2.28.0
::     3. Python 3.8.3\3.9.7\3.10.0
::     4. Visual Studio 2017\2019\2022 Community
::     5. CUDA 11.2\12\12.3  cudnn 8.1\8.9\9.6
::     6. java jre
::     7. sccache
::     8. TensorRT 8.0\8.6\10.7
::     9. xly agent
::     tools

:: Echo command is not required.
:: This script is executed in the Downloads directory

@echo off
setlocal enabledelayedexpansion

:: Step 0: wget tool
echo ">>>>>>>> step [0/9]: wget tool"
where wget > nul 2> nul || call :install_wget
goto cmake

:install_wget
echo There is not wget in this PC, will download wget 1.21.4.
echo Download package from https://eternallybored.org/misc/wget/1.21.4/64/wget.exe ...
powershell -Command "Invoke-WebRequest -Uri 'https://eternallybored.org/misc/wget/1.21.4/64/wget.exe' -OutFile 'wget.exe'"
if %errorlevel% == 0 (
  echo Download wget tool into %cd% success.
) else (
  echo Error***** Download wget tool failed, please download it before rerun.
  exit /b 1
)
copy wget.exe C:\Windows\System32\ || exit /b 1
goto :eof

:: Step 1: CMake
:cmake
echo ">>>>>>>> step [1/9]: CMake 3.18.0"
where cmake > nul 2> nul || call :install_cmake
goto git

:install_cmake
echo There is not cmake in this PC, will install cmake-3.18.0.
echo Download package from https://cmake.org/files/v3.18/cmake-3.18.0-win64-x64.msi ...
wget -O cmake-3.18.0-win64-x64.msi https://cmake.org/files/v3.18/cmake-3.18.0-win64-x64.msi
echo Install cmake-3.18.0 ...
:: /passive [silent installation]
:: /norestart [do not restart]
:: ADD_CMAKE_TO_PATH = System [add CMake to the system PATH for all users]
start /wait msiexec /i cmake-3.18.0-win64-x64.msi /passive /norestart ADD_CMAKE_TO_PATH=System
if %errorlevel% == 0 (
  echo Install CMake-3.18.0 success!
) else (
  echo Error***** Install Cmake-3.18.0 failed, please re-install it manually.
)
goto :eof

:: Step 2: Git
:git
echo ">>>>>>>> step [2/9]: Git 2.28.0"
where git > nul 2> nul || call :install_git
goto python

:install_git
echo There is not git in this PC, will install Git-2.28.0.
echo Download package from https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe ...
wget --no-check-certificate -O Git-2.28.0-64-bit.exe https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe
echo Install Git-2.28.0 ...
:: /SILENT [silent install]
:: /ALLUSERS [add path for all users]
:: /NORESTART [do not restart]
start /wait Git-2.28.0-64-bit.exe /SILENT /ALLUSERS /NORESTART
if %errorlevel% == 0 (
  echo Install Git-2.28.0 success!
) else (
  echo Error***** Install Git-2.28.0 failed, please re-install it manually.
)
goto :eof

:: Step 3: Python
:python
echo ">>>>>>>> step [3/9]: Python"
where python 2>&1 | findstr /C:"Python38" > nul 2> nul || call :install_python3.8.3
where python 2>&1 | findstr /C:"Python39" > nul 2> nul || call :install_python3.9.7
where python 2>&1 | findstr /C:"Python310" > nul 2> nul || call :install_python3.10.0
goto vs

:install_python3.8.3
echo There is not Python in this PC, will install Python-3.8.3
echo Download package from https://www.python.org/ftp/python/3.8.3/python-3.8.3-amd64.exe ...
wget --no-check-certificate -O python-3.8.3-amd64.exe https://www.python.org/ftp/python/3.8.3/python-3.8.3-amd64.exe
echo Install Python-3.8.3 ...
:: /passive [silent install]
:: InstallAllUsers [add path for all users]
:: PrependPath [add script/install into PATH]
:: TargetDir [install directory]
start /wait python-3.8.3-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python38
if %errorlevel% == 0 (
  echo Install python-3.8.3 success!
) else (
  echo Error***** Install python-3.8.3 failed, please re-install it manually.
)
goto :eof

:install_python3.9.7
echo There is not Python in this PC, will install Python-3.9.7
echo Download package from https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe ...
wget --no-check-certificate -O python-3.9.7-amd64.exe https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe
echo Install Python-3.9.7 ...
start /wait python-3.9.7-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python39
if %errorlevel% == 0 (
  echo Install python-3.9.7 success!
) else (
  echo Error***** Install python-3.9.7 failed, please re-install it manually.
)
goto :eof

:install_python3.10.0
echo There is not Python in this PC, will install Python-3.10.0
echo Download package from https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe ...
wget --no-check-certificate -O python-3.10.0-amd64.exe https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe
echo Install Python-3.10.0 ...
start /wait python-3.10.0-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python310
if %errorlevel% == 0 (
  echo Install python-3.10.0 success!
) else (
  echo Error***** Install python-3.10.0 failed, please re-install it manually.
)
goto :eof

::Windows:vs2019, cuda12.0
::Inference:vs2019, cdua11.2

:: Step 4: Visual Studio
:vs
echo ">>>>>>>> step [4/9]: Visual Studio"

if "%~1"=="" (
    echo 请提供参数1: vs2017, vs2019, vs2022
    exit /b 1
)
set VS_FLAG=%~1

if /i "%VS_FLAG%"=="vs2017" (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" (
        echo "Visual Studio 2017 already installed."
    ) else (
        call :install_visual_studio2017
    )
)
if /i "%VS_FLAG%"=="vs2019" (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        echo "Visual Studio 2019 already installed."
    ) else (
        call :install_visual_studio2019
    )
)
if /i "%VS_FLAG%"=="vs2022" (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\\vcvars64.bat" (
        echo "Visual Studio 2022 already installed."
    ) else (
        call :install_visual_studio2022
    )
)
goto cuda

:install_visual_studio2017
echo There is not Visual Studio in this PC, will install VS2017.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/VS2017/vs_Community.exe"
wget --no-check-certificate -O vs_Community.exe "https://paddle-ci.gz.bcebos.com/window_requirement/VS2017/vs_Community.exe"
:: /passive [silent install]
:: /norestart [no restart]
:: /NoRefresh [no refresh]
:: /InstallSelectableItems NativeLanguageSupport_Group [select Visual C++ for installing]
::start /wait vs_Community.exe --passive --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.Universal --includeRecommended
powershell -Command "Start-Process -FilePath 'vs_Community.exe' -ArgumentList '--passive', '--add', 'Microsoft.VisualStudio.Workload.NativeDesktop', '--add', 'Microsoft.VisualStudio.Workload.Universal', '--includeRecommended' -Wait"
if %errorlevel% == 0 (
  echo Install Visual Studio 2017 success!
) else (
  echo Error***** Install Visual Studio 2017 failed, please re-install it manually.
)
goto :eof

:install_visual_studio2019
echo There is not Visual Studio in this PC, will install VS2019.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/vs_community__2019.exe"
wget --no-check-certificate -O vs_Community.exe "https://paddle-ci.gz.bcebos.com/window_requirement/vs_community__2019.exe"
echo Install Visual Studio 2019 ...
::start /wait vs_Community.exe --passive --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.Universal --includeRecommended
powershell -Command "Start-Process -FilePath 'vs_Community.exe' -ArgumentList '--passive', '--add', 'Microsoft.VisualStudio.Workload.NativeDesktop', '--add', 'Microsoft.VisualStudio.Workload.Universal', '--includeRecommended' -Wait"
if %errorlevel% == 0 (
  echo Install Visual Studio 2019 success!
) else (
  echo Error***** Install Visual Studio 2019 failed, please re-install it manually.
)
goto :eof

:install_visual_studio2022
echo There is not Visual Studio in this PC, will install VS2022.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/VisualStudioSetup.exe"
echo Install Visual Studio 2022 ...
wget --no-check-certificate -O VisualStudioSetup.exe "https://paddle-ci.gz.bcebos.com/window_requirement/VisualStudioSetup.exe"
::start /wait VisualStudioSetup.exe --passive --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.Universal --includeRecommended
powershell -Command "Start-Process -FilePath 'VisualStudioSetup.exe' -ArgumentList '--passive', '--add', 'Microsoft.VisualStudio.Workload.NativeDesktop', '--add', 'Microsoft.VisualStudio.Workload.Universal', '--includeRecommended' -Wait"
if %errorlevel% == 0 (
  echo Install Visual Studio 2022 success!
) else (
  echo Error***** Install Visual Studio 2022 failed, please re-install it manually.
)
goto :eof

:: Step 5: CUDA
:cuda
echo ">>>>>>>> step [5/9]: CUDA "

if "%~2"=="" (
    echo 请提供参数2: cuda112, cuda12, cuda123
    exit /b 1
)
set CUDA_FLAG=%~2

if /i "%CUDA_FLAG%"=="cuda112" (
  if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo "Please install vs2019 first."
    exit /b 1
  )
  cmd /C nvcc --version 2> nul | findstr /C:"11.2" > nul 2> nul || call :install_cuda112
)
if /i "%CUDA_FLAG%"=="cuda12" (
  if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo "Please install vs2019 first."
    exit /b 1
  )
  cmd /C nvcc --version 2> nul | findstr /C:"12" > nul 2> nul || call :install_cuda112
)
if /i "%CUDA_FLAG%"=="cuda123" (
  if not exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\\vcvars64.bat" (
    echo "Please install vs2022 first."
    exit /b 1
  )
  cmd /C nvcc --version 2> nul | findstr /C:"12.3" > nul 2> nul || call :install_cuda112
)
goto java-jre

:install_cuda112
echo There is not CUDA in this PC, will install CUDA-11.2.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_11.2.0_460.89_win10.exe"
wget --no-check-certificate -O cuda_installer.exe "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_11.2.0_460.89_win10.exe"
echo Install CUDA-11.2 ...
:: -s [silent install]
start /wait cuda_installer.exe -s
if %errorlevel% == 0 (
  echo Install CUDA-11.2 success!
) else (
  echo Error***** Install CUDA-11.2 failed, please re-install it manually.
  goto :eof
)
echo Download cudnn from "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-11.2-windows-x64-v8.1.0.77.zip"
wget -O cudnn-11.2-windows-x64-v8.1.0.77.zip "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-11.2-windows-x64-v8.1.0.77.zip"
PowerShell -Command "Expand-Archive -Path 'cudnn-11.2-windows-x64-v8.1.0.77.zip' -DestinationPath '.' -Force"
xcopy /E /Y /R "cuda\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
xcopy /E /Y /R "cuda\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include"
xcopy /E /Y /R "cuda\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib"
rd /s /q cuda
goto :eof

:install_cuda12
echo There is not CUDA in this PC, will install CUDA-12.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_12.0.1_528.33_windows.exe"
wget --no-check-certificate -O cuda_installer.exe "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_12.0.1_528.33_windows.exe"
echo Install CUDA-12 ...
:: -s [silent install]
start /wait cuda_installer.exe -s
if %errorlevel% == 0 (
  echo Install CUDA-12 success!
) else (
  echo Error***** Install CUDA-12 failed, please re-install it manually.
  goto :eof
)
wget -O cudnn-windows-x86_64-8.9.1.23_cuda12-archive.zip "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-windows-x86_64-8.9.1.23_cuda12-archive.zip"
tar xf cudnn-windows-x86_64-8.9.1.23_cuda12-archive.zip
ren cudnn-windows-x86_64-8.9.1.23_cuda12-archive cuda
xcopy /E /Y /R "cuda\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12\bin"
xcopy /E /Y /R "cuda\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12\include"
xcopy /E /Y /R "cuda\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12\lib"
rd /s /q cuda
goto :eof

:install_cuda123
echo There is not CUDA in this PC, will install CUDA-12.3.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_12.3.0_545.84_windows.exe"
wget --no-check-certificate -O cuda_installer.exe "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_12.3.0_545.84_windows.exe"
echo Install CUDA-12 ...
:: -s [silent install]
start /wait cuda_installer.exe -s
if %errorlevel% == 0 (
  echo Install CUDA-12.3 success!
) else (
  echo Error***** Install CUDA-12.3 failed, please re-install it manually.
  goto :eof
)
wget -O cudnn-windows-x86_64-9.6.0.74_cuda12-archive.zip "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-windows-x86_64-9.6.0.74_cuda12-archive.zip"
tar xf cudnn-windows-x86_64-9.6.0.74_cuda12-archive.zip
ren cudnn-windows-x86_64-9.6.0.74_cuda12-archive cuda
xcopy /E /Y /R "cuda\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
xcopy /E /Y /R "cuda\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include"
xcopy /E /Y /R "cuda\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib"
rd /s /q cuda
goto :eof

:: Step 6: Java JRE
:java-jre
echo ">>>>>>>> step [6/9]: java jre"
cmd /C java -version > nul 2> nul || call :install_java
goto sccache

:install_java
echo There is not java-jre in this PC, will install java-jre.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/jre-8u261-windows-i586.exe"
wget -O jre-8u261-windows-x64.exe "https://paddle-ci.gz.bcebos.com/window_requirement/jre-8u261-windows-i586.exe"
echo Install java-jre ...
:: -s [silent install]
start /wait jre-8u261-windows-x64.exe /s
if %errorlevel% == 0 (
  echo Install java success!
) else (
  echo Error***** Install java failed, please re-install it manually.
)
del jre-8u261-windows-x64.exe
goto :eof

:: Step 7: sccache
:sccache
echo ">>>>>>>> step [7/9]: sccache"
cmd /C sccache -V > nul 2> nul || call :download_sccache
goto tensorrt

:download_sccache
echo There is not sccache in this PC, will install sccache.
echo Download package from https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe
wget -O sccache.exe "https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe"
copy sccache.exe C:\Python310 /Y
goto :eof

:: Step 8: TensorRT
:tensorrt
echo ">>>>>>>> step [8/9]: TensorRT"

set TARGET_PATH=D:\TensorRT

if not exist "%TARGET_PATH%" (
    mkdir "%TARGET_PATH%"
)

if /i "%CUDA_FLAG%"=="cuda112" (
    call :download_TensorRT_8_0_1_6
)
if /i "%CUDA_FLAG%"=="cuda12" (
    call :download_TensorRT_8_6_1_6
)
if /i "%CUDA_FLAG%"=="cuda123" (
    call :download_TensorRT_10_7_0_23
)
goto xly-agent

:download_TensorRT_8_0_1_6
echo "Downloading TensorRT 8.0.1.6..."
if not exist TensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip wget -O TensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip ^
"https://paddle-ci.gz.bcebos.com/window_requirement/TensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip"
PowerShell -Command "Expand-Archive -Path 'TensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip' -DestinationPath '.' -Force"
xcopy /E /Y /R "TensorRT-8.0.1.6\*" %TARGET_PATH%
goto :eof

:download_TensorRT_8_6_1_6
echo "Downloading TensorRT 8.6.1.6..."
if not exist TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip wget -O TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip ^
"https://paddle-ci.gz.bcebos.com/window_requirement/TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip"
PowerShell -Command "Expand-Archive -Path 'TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip' -DestinationPath '.' -Force"
xcopy /E /Y /R "TensorRT-8.6.1.6\*" %TARGET_PATH%
goto :eof

:download_TensorRT_10_7_0_23
echo "Downloading TensorRT 10.7.0.23..."
if not exist TensorRT-10.7.0.23.Windows.win10.cuda-12.6.zip wget -O TensorRT-10.7.0.23.Windows.win10.cuda-12.6.zip ^
"https://paddle-ci.gz.bcebos.com/window_requirement/TensorRT-10.7.0.23.Windows.win10.cuda-12.6.zip"
PowerShell -Command "Expand-Archive -Path 'TensorRT-10.7.0.23.Windows.win10.cuda-12.6.zip' -DestinationPath '.' -Force"
xcopy /E /Y /R "TensorRT-10.7.0.23\*" %TARGET_PATH%
goto :eof

:: Step 9: xly agent
:xly-agent
echo ">>>>>>>> step [9/9]: xly agent"
if not exist agent.jar (
    call :download_xly_agent
)
echo "Installation completed."
goto :download_tools

:download_xly_agent
echo "Downloading xly agent..."
::wget --no-check-certificate -O agent.jar "https://xly.bce.baidu.com/sa_server/agent/v1/download?version=1.2.8"
wget --no-check-certificate -O agent.jar "https://paddle-ci.gz.bcebos.com/window_requirement/agent.jar"
goto :eof

:download_tools
if not exist tools.zip (
    wget -O tools.zip "https://paddle-ci.gz.bcebos.com/window_requirement/tools.zip"
    PowerShell -Command "Expand-Archive -Path 'tools.zip' -DestinationPath '.' -Force"
    mkdir C:\home\workspace\cache\tools
    xcopy "tools\*" "C:\home\workspace\cache\tools" /E /I /H
)

if /i "%CUDA_FLAG%"=="cuda12" (
    wget -O zlipwapi123x64.zip "https://paddle-ci.gz.bcebos.com/window_requirement/zlipwapi123x64.zip"
    tar xf zlipwapi123x64.zip
    xcopy /E /Y /R  "zlipwapi\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib"
    xcopy /E /Y /R  "zlipwapi\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
)

pause
