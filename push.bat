@echo off
title Push LogitGuard-RAG to GitHub
cd /d "%~dp0"

echo ============================================================
echo Pushing LogitGuard-RAG commits to GitHub
echo Remote: https://github.com/saranxzi/LogitGuard-RAG.git
echo ============================================================
echo.

git push -u origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTE] If prompted by Git Credential Manager, complete the
    echo browser login to authorize pushing to github.com/saranxzi/LogitGuard-RAG.git
)

echo.
pause
