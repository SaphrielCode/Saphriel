@echo off
setlocal
"%~dp0build\llama-cpu\bin\Release\llama-server.exe" --model "%~dp0models\SmolLM2-135M-Instruct-f16.gguf" --gpu-layers 0 --threads 8 --threads-batch 8 --ctx-size 2048 --batch-size 128 --ubatch-size 128 --host 127.0.0.1 --port 8080 %*
exit /b %errorlevel%
