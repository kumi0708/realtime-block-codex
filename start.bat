@echo off
cd /d "%~dp0"

if not exist node_modules (
  echo node_modules が見つかりません。npm install を実行します...
  call npm install
  if errorlevel 1 (
    echo npm install に失敗しました。
    pause
    exit /b 1
  )
)

echo 開発サーバーを起動します...
start "" http://localhost:5173/
call npm run dev -- --port 5173
pause
