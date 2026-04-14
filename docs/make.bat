@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo.The 'sphinx-build' command was not found. Install Sphinx:
    echo.   pip install sparsehydro[docs]
    exit /b 1
)

if "%1" == "" goto help
if "%1" == "html" goto html
if "%1" == "clean" goto clean
goto help

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O%
echo Build finished. HTML pages are in %BUILDDIR%/html.
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:end
popd
