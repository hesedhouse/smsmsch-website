@echo off
chcp 65001 >nul
echo [%date% %time%] 설교 자동 게시 시작 >> "%~dp0local_run.log"

cd /d "%~dp0"

REM 최신 코드 pull
git pull origin master

REM 스크립트 실행
python scripts/publish_sermon.py

REM 변경사항 커밋 & 푸시
git add -A
git diff --staged --quiet
if errorlevel 1 (
    git commit -m "auto: 새 설교 말씀 게시"
    git push origin master
    echo [%date% %time%] 게시 완료 >> "%~dp0local_run.log"
) else (
    echo [%date% %time%] 새 설교 없음 >> "%~dp0local_run.log"
)
