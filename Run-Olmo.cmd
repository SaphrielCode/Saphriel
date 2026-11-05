@echo off
setlocal
if not defined LLAMA_MMAP_PREFETCH set LLAMA_MMAP_PREFETCH=0
rem Use the unchanged GGUF weights without an additional CPU repacking buffer.
"%~dp0build\llama-cpu\bin\Release\llama-cli.exe" --model "%~dp0models\Olmo-3.1-32B-Instruct-Q4_K_M.gguf" --no-repack --load-mode mmap --gpu-layers 0 --threads 8 --threads-batch 8 --ctx-size 2048 --batch-size 128 --ubatch-size 128 --predict 128 %*
exit /b %errorlevel%
