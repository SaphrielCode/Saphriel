@echo off
setlocal
"%~dp0build\llama-cpu\bin\Release\llama-cli.exe" --model "%~dp0models\SmolLM2-135M-Instruct-f16.gguf" --gpu-layers 0 --threads 8 --threads-batch 8 --ctx-size 2048 --batch-size 128 --ubatch-size 128 --predict 256 %*
exit /b %errorlevel%
